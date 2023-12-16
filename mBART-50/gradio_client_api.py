from gradio_client import Client


def get_translate1(text):
    '''
    新版本 4.x gradio
    请求接口方法
    '''
    client = Client("http://127.0.0.1:7860/")
    result = client.predict(
        text,  # str  in '请输入翻译文本' Textbox component
        api_name="translate"
    )
    return result


def get_translate2(text):
    '''
    gradio-3.50.2 gradio-client-0.6.1
    参考文档是：  https://www.gradio.app/3.50.2/guides/quickstart
    '''
    from gradio_client import Client

    client = Client("http://127.0.0.1:7860/")
    result = client.predict(
        text,
        api_name="/translate"
    )
    return result


if __name__ == '__main__':
    print(get_translate2('ValueError: Cannot find a function with `api_name`: /translate.'))
