import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from PIL import ImageFont, ImageDraw, Image
import time
from flask import Flask, render_template, Response, request
import DBManager as db

app = Flask(__name__)

# 새 얼굴이 인식되었는지 확인용 플래그
flag = False
# 찾은 얼굴의 이름과 학번을 임시 저장할 변수
tmp_sno = ""
tmp_sname = ""

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=False, device=device)

# 데이터베이스 경로취소
# database_path = 'face_database.pkl'

# 임시 저장소 및 시간 설정
temp_storage = {}
temp_storage_duration = 60  # 초 단위로 임시 저장 시간 설정

database_cache = None  # 캐싱된 데이터베이스

# 데이터베이스 로드
def load_database(refresh=False):
    global database_cache
    if database_cache is not None and not refresh:
        return database_cache  # 기존 데이터 반환
    dbms = db.DBManager()
    dbflag = dbms.DBOpen(
        host="localhost",
        dbname="facenetdb",
        id="root",
        pw="ezen"
    )
    if not dbflag:
        print("데이터베이스 연결 오류입니다")
        return {}
    
    sql = "SELECT embedding.sno, sname, embedding.embedding FROM studentinfo, embedding WHERE studentinfo.sno = embedding.sno"
    dbms.OpenQuery(sql)
    database_cache = dbms.GetDf()  # 데이터베이스 캐싱
    dbms.CloseQuery()
    dbms.DBClose()
    print("DB에서 사용자 정보를 읽어왔습니다")
    print(database_cache)
    return database_cache

# def load_database():
#     """데이터베이스 로드 함수"""
#     if os.path.exists(database_path):
#         with open(database_path, 'rb') as f:
#             return pickle.load(f)
#     return {}

# def save_database(database):
#     """데이터베이스 저장 함수"""
#     with open(database_path, 'wb') as f:
#         pickle.dump(database, f)

def name_search(phone):
    dbms = db.DBManager()
    dbflag = dbms.DBOpen(
        host="localhost",
        dbname="facenetdb",
        id="root",
        pw="ezen"
    )
    if dbflag == False :
        print("데이터베이스 연결 오류입니다")
        return {}
    else :
        sql = f"select sno, sname from studentinfo where phone = { phone }"
        dbms.OpenQuery(sql)
        sno = dbms.GetValue(0, 'sno')
        sname = dbms.GetValue(0, 'sname')
        dbms.CloseQuery()
        dbms.DBClose()
        return sno, sname

# ajax로 요청받을때 데이터를 응답하는 api 
import json
from flask import Response
@app.route("/search")
def search_ok() :
    phone = request.args.get("phone")
    # DB를 조회하고 결과를 받는다
    sno, sname = name_search(phone)
    # 결과를 HTML코드로 가공한다
    response = f'이름 : {sname} <button type="button" onclick="location.href=\'checkout?no={sno}\'">출석 확인</button>'
    return response

def get_embedding(face_img):
    """얼굴 이미지에서 임베딩 추출"""
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (160, 160))
    face_img = np.transpose(face_img, (2, 0, 1))    # (H, W, C) -> (C, H, W)
    face_img = torch.tensor(face_img, dtype=torch.float32).div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_img).cpu().numpy().flatten()
        # print(type(embedding))
    return embedding

def recognize_face(face_img, database, threshold=0.6):
    # if not database.empty : print(database); print(type(database)); print(database.iloc[0])
    global flag
    if not flag :
        """얼굴 인식 함수"""
        embedding = get_embedding(face_img)
        min_distance = float('inf')
        match_name = None
        match_no = None
        
        print("database와 비교를 시작합니다", len(database) )
        
        for i in range(len(database)):
            item = database.iloc[i]
            sno = item['sno']
            sname = item['sname']
            db_embedding = item['embedding']
            
            # 문자열을 numpy 배열로 변환
            db_embedding = np.array(list(map(float, db_embedding.split(','))))
            distance = cosine(embedding, db_embedding)
            
            if distance < min_distance:
                min_distance = distance
                match_name   = sname
                match_no     = sno
                
        if min_distance < threshold:
            #print("return : no, name")
            flag = True
            return match_no, match_name
        else:
            #print("return : None")
            return None

def save_database(new_face):
    """데이터베이스 저장 함수"""
    global flag  # 전역 변수 사용 선언
    embedding = new_face.iloc[1]
    embedding_str = ','.join(map(str, embedding))
    sno = new_face.iloc[0]
    
    # DBManager 객체 생성
    dbms = db.DBManager()
    dbflag = dbms.DBOpen(
        host="localhost",
        dbname="facenetdb",
        id="root",
        pw="ezen"
    )
    if not dbflag:
        print("데이터베이스 연결 오류입니다")
        return

    sql = "INSERT INTO embedding (embedding, sno) VALUES (%s, %s)"
    dbms.RunSQL(sql, (embedding_str, sno))
    dbms.CloseQuery()
    dbms.DBClose()

    flag = False  # 새로운 얼굴 등록 후 플래그 초기화

def is_recently_detected(embedding, temp_storage, threshold=0.6):
    """임시 저장소에서 최근에 감지된 얼굴인지 확인"""
    current_time = time.time()
    for temp_embedding, timestamp in list(temp_storage.items()):
        if cosine(embedding, temp_embedding) < threshold:
            # 최근에 감지된 얼굴이면 True 반환
            if current_time - timestamp < temp_storage_duration:
                return True
            else:
                # 시간 초과된 항목은 제거
                del temp_storage[temp_embedding]
    return False

def draw_text(img, text, position, font_path='malgun.ttf', font_size=20, color=(255, 255, 255)):
    """이미지에 텍스트 그리기"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("⚠️ 폰트 파일을 찾을 수 없습니다. 기본 폰트로 대체합니다.")
        font = ImageFont.load_default()  # 기본 폰트 사용

    draw.text(position, text, font=font, fill=color)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

# 웹캠을 열어서 찍은 영상을 이미지로 변환하는 함수
cap = None  # 전역 변수 선언 (웹캠 유지)

def get_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return None
    return cap

def gen_frames():
    global flag, tmp_sno, tmp_sname
    database = load_database()
    camera = get_camera()
    if camera is None:
        return
    
    try :
        while True:
            ret, frame = camera.read()
            if not ret:
                print("프레임을 가져올 수 없습니다.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = None
            
            if not flag or tmp_sname == "":
                boxes, _ = mtcnn.detect(rgb_frame)
            # else :
            #     print("더이상 얼굴 인식을 하지않습니다")

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face_img = frame[y1:y2, x1:x2]

                    if face_img is None or face_img.size == 0:
                        continue

                    result = recognize_face(face_img, database)
                    print("인식 결과 :\n",result)

                    if result:
                        tmp_sno, tmp_sname = result
                        label = f"{tmp_sname} ({tmp_sno})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        frame = draw_text(frame, label, (x1, y1 - 30))
                        print(f"flag : {flag} / tmp_sno : {tmp_sno} / tmp_sname : {tmp_sname} ")
                    else:
                        embedding = get_embedding(face_img)
                        #print(embedding)
                        if not is_recently_detected(embedding, temp_storage):
                            print("일치하는 얼굴을 찾지 못함")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            frame = draw_text(frame, "새로운 얼굴", (x1, y1 - 30))
                            temp_storage[tuple(embedding)] = time.time()
                            flag = True

            ret, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

    except Exception as e :
        camera.release()
        print(e)
    
# /를 요청하면 템플릿 파일중에 facerecog.html 문서를 열어줌
@app.route('/')
def index():
    global flag, tmp_sno, tmp_sname
    flag = False
    tmp_sname = ""
    tmp_sno = ""
    return render_template('facerecog.html')

# '/video_feed' 주소를 요청하면, gen_frames()함수를 실행하여 반환된 값을
# multipart 데이터로 응답함
# gen_frames()함수 내용에 안면 인식 되는 코드가 호출되면 됩니다
@app.route('/video_feed')
def video_feed():
    # return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

import json
from flask import Response
@app.route('/check_flag')
def check_flag():
    global flag, tmp_sno, tmp_sname
    data = {'flag': flag}

    if flag:
        if not tmp_sno or tmp_sno is None : tmp_sno = 0
        data.update({'sno': int(tmp_sno), 'sname': tmp_sname})
        
    print(data)
    print(json.dumps(data, ensure_ascii=False))

    return Response(json.dumps(data, ensure_ascii=False), content_type='application/json; charset=utf-8')

@app.route('/recognized')
def get_name():
    global tmp_sno, tmp_sname
    name = request.args.get("name")
    no   = request.args.get("no")
    tmp_sno = no
    tmp_sname = name
    return render_template('facerecog.html', name=name, no=no)

@app.route('/unrecognized')
def new_face_detected():
    return render_template('facerecog.html', name='new')

@app.route('/checkout')
def checkout():
    sno   = request.args.get("sno")
    dbms = db.DBManager()
    dbflag = dbms.DBOpen(
        host="localhost",
        dbname="facenetdb",
        id="root",
        pw="ezen"
    )
    if not dbflag:
        print("데이터베이스 연결 오류입니다")
        return {}
    
    sql = "INSERT INTO attendance (sno, classno) SELECT sno, classno FROM studentinfo WHERE sno =" + sno

    dbms.RunSQL(sql)
    dbms.CloseQuery()
    dbms.DBClose()
    return 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
