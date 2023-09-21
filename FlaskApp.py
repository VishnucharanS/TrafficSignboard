from flask import Flask, render_template, Response,jsonify,request,session

#from flask_wtf import FlaskForm

#from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
#from werkzeug.utils import secure_filename
#from wtforms.validators import InputRequired,NumberRange
#import os

import cv2


from Yolo_model import video_detection
app = Flask(__name__)

app.config['SECRET_KEY'] = 'vishnucharan'
app.config['UPLOAD_FOLDER'] = 'static/files'

def generate_frames(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(path_x=0),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug = True)