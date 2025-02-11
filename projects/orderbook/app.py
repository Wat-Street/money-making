import os, glob
from flask import Flask, jsonify, request
from db_config import get_db_connection
from docker_utils import build_docker_image, run_docker_container, stop_docker_container
from github_utils import recursive_repo_clone

ORDERBOOKS_TABLE_NAME = 'order_books_v2'


app = Flask(__name__)


@app.route('/create_orderbook', methods=['GET'])
def create_orderbook():
    """
    This endpoint creates an orderbook instance. It expects the following arguments:
        - name: unique orderbook name
        - tickerstotrack: tickers (e.g. (AAPL, GOOG))
        - algo_path: GitHub URL. Specific branch and filepath are supported, but optional. (e.g. 'https://github.com/Wat-Street/money-making/tree/main/projects/orderbook_test_model')
        - updatetime: time interval for updates (minutes)
        - end: lifespan of instance (days)
    It creates an entry in the database for the orderbook, as well as generates a Docker image, saved as a tar file in [TODO: directory]
    """
    name = request.args.get('name')
    tickers_to_track = request.args.get('tickerstotrack', '').split(',')
    algo_path = request.args.get('algo_path')
    update_time = request.args.get('updatetime', type=int)
    end_duration = request.args.get('end', type=int)

    # input validation
    if not name or not algo_path or not update_time or not end_duration:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # pull algorithm into local
        temp_model_store = "temporary_model_storage"
        recursive_repo_clone(algo_path, temp_model_store)
        print(f'Successfully pulled algo {name} repo to temporary model storage')

        # paths to pull algorithm and store image
        path_to_algo = f"{temp_model_store}"
        path_to_image_store = f"docker_images/{name}.tar"
        
        # check if image with this name already exists. If so, delete it first.
        # Then, build it from the path_to_algo.
        if os.path.exists(path_to_image_store):
            os.remove(path_to_image_store)
        
        # build docker image
        image = build_docker_image(name, path_to_algo)

        # save image as .tar to path_to_image_store
        with open(path_to_image_store, 'wb') as image_tar:
            for chunk in image.save():
                image_tar.write(chunk)
        print(f"Saved Docker image for '{name}' to {path_to_image_store}")
        
        # delete the temporary model storage folder after image build
        for file in glob.glob('{temp_model_store}/*'):
            os.remove(file)
        print(f'Successfully removed contents of temporary model storage')

        # save the order book in the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute( # TODO sqlalchemy
            f"""
                INSERT INTO {ORDERBOOKS_TABLE_NAME} (name, tickers_to_track, algo_link, update_time, end_duration)
                VALUES ('{name}', ARRAY {tickers_to_track}, '{algo_path}', '{update_time}', '{end_duration}')
            """
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'info': f"Order book '{name}' has been created."}), 201
    
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route("/view_orderbook", methods=["GET"])
def view_orderbook():
    """
    This endpoint allows you to view an order book.
    Expects: name of algorithm.
    Returns: a json containing the trades, worth, and balance of the order book.
    """
    name = request.args.get('name')

    # retrieve order book from database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT trades, worth, balance FROM {ORDERBOOKS_TABLE_NAME} WHERE name = '{name}'")
    result = cur.fetchone()
    conn.close()
    
    # return result if not empty, 404 otherwise
    if result:
        return jsonify({
            "trades": result[0],
            "worth": result[1],
            "balance": result[2],
        })
    return {"error": "Order book not found"}, 404


@app.route("/delete_orderbook", methods=['GET'])
def delete_orderbook():
    """
    This endpoint deletes an orderbook instance.
    Expects: name of algorithm.
    This function deletes the orderbook instance from the database. The image persists in the docker_images folder.
    """
    name = request.args.get('name')

    conn = get_db_connection()
    cur = conn.cursor()

    # check if the order book exists in the database
    cur.execute(f"SELECT name from {ORDERBOOKS_TABLE_NAME} WHERE name = '{name}'")
    result = cur.fetchone()
    print(result)
    print(f"DELETE FROM {ORDERBOOKS_TABLE_NAME} WHERE name = '{name}';")
    
    if not result:
        return {"Error": f"You are trying to delete an order book called '{name}'that does not exist."}, 404
    
    # delete the order book from the table
    
    cur.execute(f"DELETE FROM {ORDERBOOKS_TABLE_NAME} WHERE name = '{name}';")
    conn.commit()
    cur.close()
    conn.close()
    return {'Info': f"Deleted order book named '{name}'"}


if __name__ == "__main__":
    app.run()
