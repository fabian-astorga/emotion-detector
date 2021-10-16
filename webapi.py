from flask import Flask, jsonify, request
from process_image import *

app = Flask(__name__)
app.debug = True

@app.route("/process-image", methods=['POST'])
def index():
    data = request.get_json(force=True)
    image_base64 = data["image"]

    if image_base64 is None:
        return jsonify( {"response": "Could not process image. Try again.", "status":500} )
    else:
        image_response = catalog_image(image_base64)
        return jsonify({"response": str(image_response), "status":200})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3030)