import httpx
import requests
import json

from .utils import tran_jpg_binary, get_base_dir

class AllResponse:
    def __init__(self, app_id="52a3d649-8ffc-4824-bb41-8c203fb6ea1f"):
        self.url = "https://qianfan.baidubce.com/v2/app/conversation"
        self.run_url = "https://qianfan.baidubce.com/v2/app/conversation/runs"
        self.file_upload_url = "https://qianfan.baidubce.com/v2/app/conversation/file/upload"
        self.json_header = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer bce-v3/ALTAK-DOynWVfMBaSDMXJYX5b7G/149ab2715d4a1f6940d401961e7d835b59a113cc'
            # 这是必须的
        }
        self.data_header = {
            # 'Content-Type': 'multipart/form-data',
            'Authorization': 'Bearer bce-v3/ALTAK-DOynWVfMBaSDMXJYX5b7G/149ab2715d4a1f6940d401961e7d835b59a113cc'
        }
        self.app_id = app_id

        self.cv_id = self._create_conversation_id()
    def _create_conversation_id(self):
        payload = json.dumps({
            "app_id": self.app_id,
            # "query": "你好"
        }, ensure_ascii=False)

        response = requests.request("POST", self.url, headers=self.json_header, data=payload.encode("utf-8"))
        return response.json()["conversation_id"]
    def get_response(self, query, location, stream=False):
        payload1 = json.dumps({
            "app_id": self.app_id,
            "query": location + "<split>" + query,
            "conversation_id": self.cv_id,
            "stream": stream,
        }, ensure_ascii=False)
        res = requests.request("POST", self.run_url, headers=self.json_header, data=payload1.encode("utf-8"))
        return res
    async def get_response_async(self, query, location, stream=False):
        payload1 = json.dumps({
            "app_id": self.app_id,
            "query": location + "<split>" + query,
            "conversation_id": self.cv_id,
            "stream": stream,

        }, ensure_ascii=False)

        async with httpx.AsyncClient() as client:
            res = await client.post(
                self.run_url,
                headers=self.json_header,
                content=payload1.encode("utf-8")
            )
            return res

class FileResponse:
    """file_path只需要输入文件名即可，所有pic都需要存入pic文件夹下统一转换读取，file"""
    def __init__(self, Response, file_path=None, file=None):
        self.response = Response
        self.file_path = file_path
        self.file = file
        self.file_id = self._create_file_id()

    def _create_file_id(self):
        if self.file_path is not None:
            b_file = tran_jpg_binary(self.file_path)
        if self.file is not None:
            b_file = self.file
        # b_file = open(get_dir_name() + "/data/pic/" + self.file_path, 'rb')
        payload = {
            # 'app_id': self.app_id,
            'app_id': self.response.app_id,
            'conversation_id': self.response.cv_id
        }
        file = [
            ("file", ("upload.jpg", b_file, 'image/jpeg'))
        ]
        response = requests.request("POST", self.response.file_upload_url, headers=self.response.data_header, data=payload, files=file)
        return response.json()["id"]


    def get_response(self, query, location, stream=False):
        payload1 = json.dumps({
            "app_id": self.response.app_id,
            "query": location + "<split>" + query,
            "conversation_id": self.response.cv_id,
            "stream": stream,
            "file_ids": [
                self.file_id
            ]
        }, ensure_ascii=False)

        res = requests.request("POST", self.response.run_url, headers=self.response.json_header, data=payload1.encode("utf-8"))

        return res

    async def get_response_async(self, query, location, stream=False):
        payload1 = json.dumps({
            "app_id": self.response.app_id,
            "query": location + "<split>" + query,
            "conversation_id": self.response.cv_id,
            "stream": stream,
            "file_ids": [
                self.file_id
            ]
        }, ensure_ascii=False)

        async with httpx.AsyncClient() as client:
            res = await client.post(
                self.response.run_url,
                headers=self.response.json_header,
                content=payload1.encode("utf-8")
            )
            return res

def get_weather(city_name):
    url = "http://appbuilder.baidu.com/v2/components/c-wf-933c4600-dad6-4349-8b62-2c2c4abedbdb"
    headers = {
        "Content-Type": "application/json",
        'Authorization': 'Bearer bce-v3/ALTAK-DOynWVfMBaSDMXJYX5b7G/149ab2715d4a1f6940d401961e7d835b59a113cc'
    }
    payload = {
        "stream": False,
        "parameters": {"_sys_origin_query": city_name},
    }

    response = requests.post(url, json=payload, headers=headers)
    # print(response.json()["content"][0]["text"]["data"])
    return response.json()["content"][0]["text"]["data"].encode('gbk', errors='ignore').decode('gbk')

if __name__ == '__main__':
    all_response = AllResponse()
    file_response = FileResponse(all_response, "test.jpg")
    file_response.get_response("这是什么", "福建省泉州市安溪")
