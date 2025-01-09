import os
from flask import Flask, jsonify, request
from db_config import get_db_connection


app = Flask(__name__)

"""
This endpoint creates an orderbook instance. It expects the following arguments:
    - name: unique name of algorithm
    - tickerstotrack: two tickers (e.g. (AAPL, GOOG))
    - algo_path: path to algorithm from the projects directory (ex. harv-extension')
    - updatetime: time interval for updates (minutes)
    - end: lifespan of instance (days)
It creates an entry in the database for the orderbook, as well as generates a Docker image, saved as a tar file in [TODO: directory]
"""
@app.route('/create_orderbook', methods=['GET'])
def create_orderbook():
    name = request.args.get('name')
    tickers_to_track = request.args.get('tickerstotrack', '').split(',')
    algo_path = request.args.get('algo_path')
    update_time = request.args.get('updatetime', type=int)
    end = request.args.get('end', type=int)

    # input validation
    if not name or not algo_path or not update_time or not end:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # paths to pull algorithm and store image
    path_to_algo = f"../{algo_path}"
    path_to_image = f"docker_images/{name}.tar"
    
    # check if image with this name already exists
    if os.path.exists(path_to_image):
        return jsonify({'Error': "You've already built this image"}), 400
    
    # build docker image
    image = build_docker_image()




def get_orderbook(name):
    conn = psycopg2.connect(f'dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}')
    cur = conn.cursor()
    cur.execute("SELECT trades, worth, balance FROM order_books WHERE name = %s", (name,))
    result = cur.fetchone()
    conn.close()
    
    if result:
        return {
            "trades": result[0],
            "worth": result[1],
            "balance": result[2],
        }
    return {"error": "Order book not found"}

@app.route("/view_orderbook/<name>", methods=["GET"])
def view_orderbook(name):
    orderbook = get_orderbook(name)
    return jsonify(orderbook)

if __name__ == "__main__":
    app.run()