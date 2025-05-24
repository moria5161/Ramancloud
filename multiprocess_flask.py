from flask import Flask, request, jsonify
from multiprocessing import Pool
import numpy as np
from api.airPLS import ZhangFit
from api.modpoly import mod_poly, imod_poly
from api.PEER import peer

app = Flask(__name__)

def airPLS_parallel_process(data, lambda_, order_):
    with Pool(processes=8) as pool:
        res = pool.starmap(ZhangFit, [(row, lambda_, order_) for row in data])
        print(len(data[1,:]))
    return np.array(res)

def ModPoly_parallel_process(data, degree, gradient, repitition):
    with Pool() as pool:
        results = pool.starmap(mod_poly, [(row, degree, repitition, gradient, idx) for idx, row in enumerate(data)])
    
    results.sort(key=lambda x: x[1])
    corrected_data = [result[0] for result in results]
    return np.array(corrected_data)

def IModPoly_parallel_process(data, degree, gradient, repitition):
    with Pool() as pool:
        results = pool.starmap(imod_poly, [(row, degree, repitition, gradient, idx) for idx, row in enumerate(data)])
    
    # Maintain original row order
    results.sort(key=lambda x: x[1])
    corrected_data = [result[0] for result in results]
    return np.array(corrected_data)

def PEER_parallel_process(data, loops, hlaf_k_threshold):
    with Pool(processes= 8) as pool:
        res = pool.starmap(peer, [(row, loops, hlaf_k_threshold) for row in data])
    return np.array(res)


@app.route('/airPLS', methods=['POST'])
def airPLS_handler():
    try:
        # 获取 JSON 请求数据
        req_data = request.json
        x = np.array(req_data['data'])
        lambda_ = req_data['lambda']
        order_ = req_data['order']
        
        size = x.shape
        res = airPLS_parallel_process(x.reshape(-1, size[-1]), lambda_, order_)
        res = res.reshape(size)

        # 将结果转换为列表并返回 JSON 响应
        return jsonify(res.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.route('/modpoly', methods=['POST'])
def modpoly_handler():
    try:
        req_data = request.json
        x = np.array(req_data['data'])
        order_ = req_data['order']
        gradient = req_data['gradient']
        repitition = req_data['repitition']

        size = x.shape
        res = ModPoly_parallel_process(x.reshape(-1, size[-1]), order_, gradient, repitition)
        res = res.reshape(size)

        return jsonify(res.tolist())
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/imodpoly', methods=['POST'])
def imodpoly_handler():
    try:
        req_data = request.json
        x = np.array(req_data['data'])
        print(x.shape)
        print(x.shape[0])
        if x is None:
            print("x is None")
        order_ = req_data['order']
        if order_ is None:
            print("order_ is None")
        gradient = req_data['gradient']
        if gradient is None:
            print("gradient is None")
        repitition = req_data['repitition']
        if repitition is None:
            print("repitition is None")

        size = x.shape
        res = IModPoly_parallel_process(x.reshape(-1, size[-1]), order_, gradient, repitition)
        res = res.reshape(size)

        return jsonify(res.tolist())
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    
@app.route('/PEER', methods=['POST'])
def PEER_handler():
    try:
        # 获取 JSON 请求数据
        req_data = request.json
        x = np.array(req_data['data'])
        loops = req_data['loops']
        hlaf_k_threshold = req_data['hlaf_k_threshold']

        size = x.shape
        res = PEER_parallel_process(x.reshape(-1, size[-1]), loops, hlaf_k_threshold)
        res = res.reshape(size)
        
        # 将结果转换为列表并返回 JSON 响应
        return jsonify(res.tolist())
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
