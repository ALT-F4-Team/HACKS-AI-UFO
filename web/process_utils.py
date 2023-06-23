def predict(model, text, idx2labels):
    text = [i for i in text.split('. ')]

    ans = {
        'Требования': [],
        'Условия': [],
        'Обязанности': []
    }

    for line in text:
        idx = model.predict(line, idx2labels)
        ans[idx].append(line)

    return ans
