from flask import Flask, jsonify
import psycopg2

app = Flask(__name__)

def make_orderbook(name, tickers, )



def get_orderbook(name):
    conn = psycopg2.connect('dbname=postgres user=reebxu password=watstreet')
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