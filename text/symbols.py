""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
# _pad        = '_'
# _punctuation = ';:,.!?¡¿—…"«»“” '
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# For Chinese
_pad        = '_'
_punctuation = '~;:,.!?¡¿—…"«»“” '
_letters = 'abcdefghijklmnopqrstuvwxyz1234'
_letters_ipa = ""

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")
