import os
from flask import Flask, jsonify, request
from db_config import get_db_connection
from docker_utils import build_docker_image, run_docker_container, stop_docker_container

ORDERBOOKS_TABLE_NAME = 'order_books'


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
    end_duration = request.args.get('end', type=int)

    # input validation
    if not name or not algo_path or not update_time or not end_duration:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # paths to pull algorithm and store image
        path_to_algo = f"../{algo_path}"
        path_to_image = f"docker_images/{name}.tar"
        
        # check if image with this name already exists. If not, build it from the path_to_algo.
        if not os.path.exists(path_to_image):
            # build docker image
            image = build_docker_image(name, path_to_algo)

            # save image as .tar to path_to_image
            with open(path_to_image, 'wb') as image_tar:
                for chunk in image.save():
                    image_tar.write(chunk)
            print(f'Saved Docker image for {name} to {path_to_image}')

        # save the order book in the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            f"""
                INSERT INTO {ORDERBOOKS_TABLE_NAME} (name, tickers_to_track, algo_link, update_time, end_duration)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, tickers_to_track, algo_path, update_time, end_duration)
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'info': f"Order book '{name}' has been created."}), 201
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



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