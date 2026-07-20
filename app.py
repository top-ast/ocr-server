from flask import Flask, request, jsonify
from rapidocr_onnxruntime import RapidOCR
import os

app = Flask(__name__)

# 初始化引擎（首次启动自动下载模型，约 50MB）
print("正在加载 OCR 引擎...")
engine = RapidOCR()
print("OCR 引擎加载完成")


@app.route("/devapi/ocrOnnxFile", methods=["POST"])
def ocr_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"data": [], "error": "no file"})

    tmp_path = "/tmp/ocr_img.png"
    file.save(tmp_path)

    try:
        result, _ = engine(tmp_path)
        data = []
        if result:
            for box, text, conf in result:
                x, y = box[0]
                w = box[2][0] - box[0][0]
                h = box[2][1] - box[0][1]
                data.append({
                    "label": text,
                    "confidence": round(float(conf), 4),
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
        return jsonify({"data": data})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/")
def index():
    return "OCR Service Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
