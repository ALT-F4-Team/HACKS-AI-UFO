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
    pattern = 'QWERTYUIOPLKJHGFDSAZXCVBNMЙЦУКЕНГШЩЗХЪЭЖДЛОРПАВЫФЯЧЁБбСМИТЬйцукенгшщзхъэждлорпаёвыфячсмитьбюqwertyuioplkjhgfdsazxcvbnm1234567890/-;.,·!? '

    text = ''.join(list(filter(lambda x: x in pattern, text)))

    # text = re.split(r' [a-zA-Zа-яА-Я0-9]{2} *[!?\-.;].', text)

    text = text.replace('Обязанности', 'split').replace('Условия', 'split').replace('Должностные обязанности', 'split').replace('Обязанности', 'split').replace('Требования', 'split')
    text = re.sub(r'(?<=[;•.!?])', 'split', text)
    text = text.split('split')
    text = [i for i in text if len(i.split()) > 1]

    text = [i[i.find(re.search(r'[a-zA-Zа-яА-Я0-9]', i).group(0)):] for i in text]

    return text
