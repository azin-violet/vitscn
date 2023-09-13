import re
import cn2an
from pypinyin import lazy_pinyin, Style
import jieba
import zhon
from text.symbols import symbols

_puntuation_map = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('。', '.'),
    ('｡', '.'),
    ('．', ''),
    ('？', '?'),
    ('！', '!'),
    ('，', ','),
]]


def number_to_chinese(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text

def remove_non_stop_punctuation(text):
    text = re.sub('[%s]' % zhon.hanzi.non_stops, '', text)
    return text
    
def map_stop_puntuation(text):
    for regex, replacement in _puntuation_map:
        text = re.sub(regex, replacement, text)
    return text

def chinese_to_pinyin(text):
    text = map_stop_puntuation(text)
    text = number_to_chinese(text)
    text = remove_non_stop_punctuation(text)
    words = jieba.lcut(text, cut_all=False)
    text = ''
    for word in words:
        if not re.search('[\u4e00-\u9fff]', word):
            if word in ''.join(symbols):
                text += word
            continue
        pinyin = lazy_pinyin(word, Style.TONE3)
        if text != '':
            text += ' '
        text += ''.join(pinyin)
    return text
