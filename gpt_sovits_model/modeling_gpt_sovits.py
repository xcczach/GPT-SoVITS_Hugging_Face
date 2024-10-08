from transformers import PreTrainedModel
from .configuration_gpt_sovits import GPTSoVITSConfig

import os
import re
import LangSegment
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .t2s_lightning_module import \
    Text2SemanticLightningModule
from . import cnhubert
from .mel_processing import spectrogram_torch
# from io import BytesIO
from .models import SynthesizerTrn
from .my_utils import load_audio
from .symbols import cleaned_text_to_sequence
from .cleaner import clean_text

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

dict_language = {
    "中文": "all_zh",#全部按中文识别
    "英文": "en",#全部按英文识别#######不变
    "日文": "all_ja",#全部按日文识别
    "中英混合": "zh",#按中英混合识别####不变
    "日英混合": "ja",#按日英混合识别####不变
    "多语种混合": "auto",#多语种启动切分识别语种
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "all_zh": "all_zh", #手动添加，以防万一
    "all_ja": "all_ja", #手动添加，以防万一
    "auto": "auto" #手动添加，以防万一
}

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}  # 不考虑省略号

def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist)-1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i-1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i+1]:
            textlist[i] += textlist[i+1]
            del textlist[i+1]
            del langlist[i+1]
        else:
            i += 1

    return textlist, langlist

def clean_text_inf(text, language):
    formattext = ""
    language = language.replace("all_","")
    for tmp in LangSegment.getTexts(text):
        if language == "ja":
            if tmp["lang"] == language or tmp["lang"] == "zh":
                formattext += tmp["text"] + " "
            continue
        if tmp["lang"] == language:
            formattext += tmp["text"] + " "
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    phones, word2ph, norm_text = clean_text(formattext, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text



def nonen_clean_text_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    #【日志】 print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

# ====== 对输入文本进行切割 =========

def split(todo_text):
    """
    将大段文本按标点切割，并将每段文本(保留末尾标点)组成列表。
    """
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    """
    第一种文本分段法：基于重写的split分割后，凑4段语句推理一次。
    """
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    """
    第二种文本分段法：基于重写split分割后，凑50个字推理一次。
    """
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return [inp]
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    if len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)

def cut3(inp):
    """
    第三种文本分段法：仅仅按中文句号分割。
    """
    inp = inp.strip("\n")
    return "\n".join(["%s。" % item for item in inp.strip("。").split("。")])

# 新增两种切法

def cut4(inp):
    """
    "按英文句号.切"
    """
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    """
    "按标点符号切"
    """
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt


def load_model(cnhubert_base_path, bert_path, dict_s1, dict_s2, is_half, device):
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
    if is_half:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"

    config = dict_s1["config"]
    ssl_model = cnhubert.get_model()
    ssl_model.load(cnhubert_base_path)
    if is_half:
        ssl_model = ssl_model.half().to(device)
    else:
        ssl_model = ssl_model.to(device)

    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    # 超长日志输出-missing_keys
    vq_model.load_state_dict(dict_s2["weight"], strict=False)

    t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()

    return tokenizer, bert_model, hps, config, ssl_model, vq_model, t2s_model

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                            hps.data.win_length, center=False)
    return spec

class GPTSoVITSModel(PreTrainedModel):
    config_class = GPTSoVITSConfig

    def __init__(self, config: GPTSoVITSConfig):
        super().__init__(config)
        self.prompt_language = config.prompt_language
        device = config.device
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device = device
        self.is_half = config.is_half  # 半精度推理
        self.dtype=torch.float16 if self.is_half == True else torch.float32 #【补】

        self.ssl_model = cnhubert.CNHubert(config._hubert_config_dict, config._hubert_extractor_config_dict)
        self.bert_model = AutoModelForMaskedLM.from_config(config._bert_config_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path, trust_remote_code=True)
        self.hps = DictToAttrRecursive(config._hps_dict)
        self.hps.model.semantic_frame_rate = "25hz"
        self.gpt_config = config._gpt_config_dict
        self.vq_model = SynthesizerTrn(
        self.hps.data.filter_length // 2 + 1,
        self.hps.train.segment_size // self.hps.data.hop_length,
        n_speakers=self.hps.data.n_speakers,
        **self.hps.model)
        self.t2s_model = Text2SemanticLightningModule(self.gpt_config, "ojbk", is_train=False)
        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(device)
            self.bert_model = self.bert_model.half().to(device)
            self.vq_model = self.vq_model.half().to(device)
            self.t2s_model = self.t2s_model.half().to(device)
        else:
            self.ssl_model = self.ssl_model.to(device)
            self.bert_model = self.bert_model.to(device)
            self.vq_model = self.vq_model.to(device)
            self.t2s_model = self.t2s_model.to(device)
        self.vq_model.eval()
        self.ssl_model.eval()
        self.t2s_model.eval()



    def get_cleaned_text_final(self,text,language):
        """
        根据语言类型选择适当的文本清洗函数，并返回处理后的音素序列、单词到音素的映射以及规范化文本。
        -> phones,word2ph,norm_text
            - clean_text_inf 针对单一语种{"en","all_zh","all_ja"}
                - clean_text 和 cleaned_text_to_sequence 来自内部text模块cleaner和__init__
            - nonen_clean_text_inf 针对混合语种{"zh", "ja","auto"}
                - splite_en_inf
        """
        if language in {"en","all_zh","all_ja"}:
            phones, word2ph, norm_text = clean_text_inf(text, language)
        elif language in {"zh", "ja","auto"}:
            phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
        return phones, word2ph, norm_text
    
    def get_bert_inf(self, phones, word2ph, norm_text, language):
        device = self.device # 【补】
        is_half = self.is_half # 【补】
        
        language=language.replace("all_","")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)

        return bert
    
    def nonen_get_bert_inf(self, text, language):
        if(language!="auto"):
            textlist, langlist = splite_en_inf(text, language)
        else:
            textlist=[]
            langlist=[]
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        bert_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)

        return bert
    
    def get_bert_feature(self, text, word2ph):

        is_half = self.is_half # 【补】
        device = self.device # 【补】

        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        if(is_half==True):phone_level_feature=phone_level_feature.half()
        
        return phone_level_feature.T

    def get_bert_final(self,phones, word2ph, text,language):
        """
        根据语言 选择调用不同的函数来得到一个bert表示。
        需要输入Get_clean_text_final得到的文字素材
        -> bert
            - get_bert_inf 针对纯英文”en”
            - nonen_get_bert_inf 针对混合语种{"zh", "ja","auto"}
            - get_bert_feature 针对纯中文”all_zh”
        """
        device = self.device # 【补】

        if language == "en":
            bert = self.get_bert_inf(phones, word2ph, text, language) # 【补】
        elif language in {"zh", "ja","auto"}:
            bert = self.nonen_get_bert_inf(text, language)
        elif language == "all_zh":
            bert = self.get_bert_feature(text, word2ph).to(device)
        else:
            bert = torch.zeros((1024, len(phones))).to(device)
        return bert