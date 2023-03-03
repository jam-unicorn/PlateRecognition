import requests
import PySimpleGUI as sg

layout = [[sg.Text('请输入服务器地址：')],
          [sg.InputText(key='address', default_text='http://127.0.0.1:8000')],
          [sg.FileBrowse('选择图片', target='text', file_types=(('JPG Files', '*.jpg*'),
                                                            ('JPEG Files',
                                                             '*.jpeg*'),
                                                            ('PNG Files', '*.png*')))],
          [sg.Text('你选择的文件是:'), sg.Text('', key='text', size=(50, 1))],
          [sg.Button('开始识别', key='start')]]

window = sg.Window('Window Title', layout)

while True:
    event, values = window.read()

    address = window['address'].get()
    filename = window['text'].get()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == 'start':
        if not address:
            sg.popup('未填写服务器地址！')
        elif not filename:
            sg.popup('未上传图片！')
        else:
            res = requests.post(address, files={'file': open(filename, 'rb')})
            if res.status_code == 200:
                sg.popup('识别结果：', res.json()['result'])
            else:
                sg.popup('识别失败🤗')

window.close()
