from spellchecker import SpellChecker #pip install pyspellchecker

def spell_checker_word(token_list):
    #I - List of words or Tokens
    #O - List of Word or Tokens

    spell = SpellChecker()
    misspelled = spell.unknown(token_list)
    correct = []
    for token in token_list:
        if spell.correction(token):
            correct.append(spell.correction(token))
        else:
            correct.append(token)
    return correct
