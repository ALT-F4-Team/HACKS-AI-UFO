import re


def predict(model, text, idx2labels):
    model.model.eval()

    ans = {
        'Требования': [],
        'Условия': [],
        'Обязанности': []
    }

    for line in text:
        idx = model.predict(line, idx2labels)
        ans[idx].append(line)

    return ans


def clear_text(text):
    pattern = 'QWERTYUIOPLKJHGFDSAZXCVBNMЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧЁСМИТЬйцукенгшщзхъэждлорпаёвыфячсмитьбюqwertyuioplkjhgfdsazxcvbnm1234567890/-;.,·\n!? '

    text = ''.join(list(filter(lambda x: x in pattern, text)))
    text = text.replace('т.д.', ' ').replace('т.п.', ' ').replace('т.ч.', ' ').replace('Обязанности', '')

    text = text.replace('.', 'split').replace(';', 'split').replace(' - ', 'split').split('split')

    return text
