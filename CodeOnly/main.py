import math
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numba
from numba import cuda
from scipy.io import loadmat
@cuda.jit
def ufe_kernel(data, result):
    l = data.shape[0]  # Number of columns
    #col_i, col_j = cuda.grid(2)
    col_i = int(cuda.blockIdx.x)
    col_j = int(cuda.blockIdx.y)
    if col_i >= col_j or col_i >= l or col_j >= l:  # Skip lower triangle & diagonal
        return
    N = data.shape[1]
    tid = int(cuda.threadIdx.x)
    block_size = int(cuda.blockDim.x)
    # Shared memory for sum calculations
    shared_f11 = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_f00 = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_f10 = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_f01 = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_f11[tid] = 0
    shared_f00[tid] = 0
    shared_f10[tid] = 0
    shared_f01[tid] = 0
    cuda.syncthreads()
    # Local thread sums
    f11, f00, f10,f01 = 0, 0, 0,0

    # Striding loop to sum over all rows

    for row in range(tid, N, block_size):
        xi = data[col_i,row]
        yi = data[col_j,row]
        f11 += xi and yi
        f00 += (not xi) and (not yi)
        f10 += (xi) and (not yi)
        f01 += (not xi) and (yi)
        #debug[col_i,row] = tid
        #debug[col_j,row] = tid


    # Store partial sums in shared memory
    shared_f11[tid] = f11
    shared_f00[tid] = f00
    shared_f10[tid] = f10
    shared_f01[tid] = f01
    cuda.syncthreads()
    # Parallel reduction
    stride = block_size // 2
    while stride > 0:
        if tid < stride and tid + stride < block_size:
            shared_f11[tid] += shared_f11[tid + stride]
            shared_f00[tid] += shared_f00[tid + stride]
            shared_f10[tid] += shared_f10[tid + stride]
            shared_f01[tid] += shared_f01[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Store final result from first thread
    if tid == 0:
        final_f11 = shared_f11[0]/N
        final_f00 = shared_f00[0]/N
        final_f10 = shared_f10[0]/N
        final_f01 = shared_f01[0]/N
        if 0 < final_f11 < 1 and 0 < final_f00 < 1 and 0 < final_f10 < 1 and 0 < final_f01 < 1 and final_f01 * final_f10 != final_f00 ** 2:
            result[col_i, col_j] = round(1 - (
                            (math.log(final_f11 / final_f00)) / (math.log((final_f01 * final_f10) / (final_f00 ** 2)))),4)
            result[col_j, col_i] = result[col_i, col_j]
        else:
            result[col_i, col_j] = 0
            result[col_j, col_i] = 0


@cuda.jit
def ufe0_kernel(data, conn_matrix, results):
    l = data.shape[0]  # Number of columns
    row_count = data.shape[1]  # Number of rows

    i = int(cuda.blockIdx.x)
    j = int(cuda.blockIdx.y)
    if i >= l or j >= l or i >= j or not conn_matrix[i, j]:
        return

    tx = cuda.threadIdx.x
    # Shared memory allocation for reduction
    shared_f11 = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_f00 = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_f10 = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_f01 = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_valid = cuda.shared.array(shape=(threads_per_block,), dtype=numba.uint32)
    shared_neighbour_amount = cuda.shared.array(shape = 1, dtype=numba.int32)
    shared_neighbours = cuda.shared.array(shape = data_length , dtype=numba.uint16)
    shared_f11[tx] = 0
    shared_f00[tx] = 0
    shared_f10[tx] = 0
    shared_f01[tx] = 0
    shared_valid[tx] = 0
    cuda.syncthreads()
    results[i, j] = float("inf")

    block_size = int(cuda.blockDim.x)

    # Neighbour search

    if tx == 0:
        shared_neighbour_amount[0] = 0
    cuda.syncthreads()
   # Check when l < threads per block Да и в целом, проврить надо
    for k in range(tx,l,block_size):
        if k != j and conn_matrix[i, k]:
            neighbour_id = cuda.atomic.add(shared_neighbour_amount,0,1)
            shared_neighbours[neighbour_id] = k
        if k != i and conn_matrix[j, k] and not conn_matrix[i,k]:
            neighbour_id = cuda.atomic.add(shared_neighbour_amount,0,1)
            shared_neighbours[neighbour_id] = k
    cuda.syncthreads()

    # Iterate over rows in a strided manner

    for k in shared_neighbours[0:shared_neighbour_amount[0]]:
        # Initialize accumulators
        f11, f00, f10, f01 = 0, 0, 0, 0
        valid_count = 0
        for row in range(tx,row_count,block_size):
            if not data[k,row]:
                fi,fj = data[i,row], data[j,row]
                f11 += fi and fj
                f00 += (not fi) and (not fj)
                f10 += (fi) and (not fj)
                f01 += (not fi) and (fj)
                valid_count += 1

        # Store into shared memory
        shared_f11[tx] = f11
        shared_f00[tx] = f00
        shared_f10[tx] = f10
        shared_f01[tx] = f01
        shared_valid[tx] = valid_count
        cuda.syncthreads()

        # Parallel reduction to compute sums
        stride = block_size // 2
        while stride > 0:
            if tx < stride:
                shared_f11[tx] += shared_f11[tx + stride]
                shared_f00[tx] += shared_f00[tx + stride]
                shared_f10[tx] += shared_f10[tx + stride]
                shared_f01[tx] += shared_f01[tx + stride]
                shared_valid[tx] += shared_valid[tx + stride]
            cuda.syncthreads()
            stride //= 2

        # Compute Pearson coefficient in thread 0
        if tx == 0:
            #debug[i,j] = k
            if shared_valid[tx] > 0:
                final_f11 = shared_f11[0] / shared_valid[tx]
                final_f00 = shared_f00[0] / shared_valid[tx]
                final_f10 = shared_f10[0] / shared_valid[tx]
                final_f01 = shared_f01[0] / shared_valid[tx]

                if 0 < final_f11 < 1 and 0 < final_f00 < 1 and 0 < final_f10 < 1 and 0 < final_f01 < 1 and final_f01 * final_f10 != final_f00 ** 2: # Проблема с отрицательными значениями никуда не делась
                    results[i, j] = round(min((1 - (
                            (math.log(final_f11 / final_f00)) / (math.log((final_f01 * final_f10) / (final_f00 ** 2))))),(results[i,j])), 4)
                    results[j, i] = results[i, j]
                    # if i == 5 and j == 16:
                    #     debug[k] = 1 - (
                    #         (math.log(final_f11 / final_f00)) / (math.log((final_f01 * final_f10) / (final_f00 ** 2))))
                else:
                    #results[i, j] = min(0,results[i,j])
                    results[i, j] = 0
                    results[j, i] = 0
                    return

            else:
                results[i, j] = 0
                results[j, i] = 0
                return


def array_subdivV3(arrayFull: np.ndarray,border_marker:list, totake:list):
    rng = np.random.default_rng()
    random.seed(1)
    sumarray = np.sum(arrayFull, axis=0)
    mask1 = np.multiply(border_marker[0] <= sumarray, sumarray <= border_marker[1])
    mask2 = np.multiply(border_marker[1] < sumarray, sumarray <= border_marker[2])
    #mask3 = np.multiply(border_marker[2] < sumarray, sumarray <= 80)
    mask3 = border_marker[2] < sumarray
    print(sum(mask1))
    print(sum(mask2))
    #print(sum(mask3))
    group1 = arrayFull[:, mask1].copy()
    group2 = arrayFull[:, mask2].copy()
    group3 = arrayFull[:, mask3].copy()
    fromwhat1 = range(np.sum(mask1))
    howmuch1 = totake[0] * 200
    fromwhat2 = range(np.sum(mask2))
    howmuch2 = totake[1] * 200
    randomsample1 = rng.choice(fromwhat1,howmuch1,replace=True)
    randomsample2 = rng.choice(fromwhat2, howmuch2, replace=True)
    endingarray1 = group1[:,randomsample1]
    endingarray2 = group2[:,randomsample2]
    group12 = np.hstack([endingarray1, endingarray2])
    return group12


def find_links_UFE_GPU(arrayFull: np.ndarray) -> tuple:
    #cuda.select_device(0)
    # Stepping approach (currently deleted)
    # Actual calculations START
    arraybool = arrayFull > 0
    UFE = np.zeros((arraybool.shape[0], arraybool.shape[0]),dtype=np.float64)
    UFE0 = np.zeros((arraybool.shape[0], arraybool.shape[0]),dtype=np.float64)
    """GPU VER"""
    # Allocate GPU memory
    d_data = cuda.to_device(arraybool)
    resarr = np.zeros((arraybool.shape[0], arraybool.shape[0]),dtype=np.float64)
    res0arr = np.zeros((arraybool.shape[0], arraybool.shape[0]),dtype=np.float64)
    #d_results = cuda.device_array((arraybool.shape[0], arraybool.shape[0]), dtype=np.float64)
    #d_resulrufe0 = cuda.device_array((arraybool.shape[0], arraybool.shape[0]), dtype=np.float64)
    d_results = cuda.to_device(resarr)
    d_resulrufe0 = cuda.to_device(res0arr)
   # d_debug = cuda.to_device(np.ones((arraybool.shape[0], 1)) * 2)
    device = cuda.get_current_device()
    blocks_per_grid = arrayFull.shape[0]

    ufe_kernel[(blocks_per_grid, blocks_per_grid), threads_per_block](d_data, d_results)
    cuda.synchronize()
    UFE[:, :] = d_results.copy_to_host()  # Ноль ли UFE?
    ufe_mask = UFE >= 0.4  # Параметр 1
    d_conn_matrix = cuda.to_device(ufe_mask)
    ufe0_kernel[(blocks_per_grid, blocks_per_grid), threads_per_block](d_data, d_conn_matrix, d_resulrufe0)
    cuda.synchronize()
    UFE0[:,:] = d_resulrufe0.copy_to_host()
    UFE0div = np.zeros(UFE.shape)
    for i2 in range(UFE0div.shape[0]):
        for j2 in range(UFE0div.shape[1]):
            if UFE[i2, j2] != 0:
                UFE0div[i2, j2] = (UFE0[i2, j2] / UFE[i2, j2])
    UFEwZeroCut = UFE0div >= 0.5  # Параметр 2
    d_results[:] = 0.0
    d_data[:] = 0
    d_resulrufe0[:] = 0.0

    # Actual calculations END
    # del d_data
    # del d_results
    # del d_resulrufe0
    # del d_debug
    # del d_conn_matrix
    # gc.collect()
    # cuda.current_context().deallocations.clear()


    return (UFEwZeroCut, UFE, UFE0)


def draw_connections_cluster(labels: np.ndarray, data: np.ndarray, UFE) -> None:
    dataname = "20k"
    GUFE = nx.from_numpy_array(data)  # with all cutoffs
    isolatedNodes = list(nx.isolates(GUFE))

    # currentlabels = labels[i*step:i*step+windowsize]
    # print(i)
    # print(i*step)
    # print(i*step+windowsize)
    currentlabels = labels

    # print(currentlabels)
    labelsd = dict(zip(list(GUFE), currentlabels))
    # edge_labels = dict([((n1, n2), f'E = {UFE[n1, n2]}')
    #                   for n1, n2 in GUFE.edges])
    print(labelsd)
    # print(edge_labels)
    options = {
        'node_size': 100,
        'width': 3,
        'edge_color': 'black',
        # 'labels': labelsd,
        'font_size': 120,
        'font_color': "black"
    }

    # complex drawing

    alreadyDraw = []
    iterator = 0
    print(f"isolates: {isolatedNodes}")
    for node in GUFE:
        # print(f"current node: {node}")
        # print(f"already drawn nodes: {alreadyDraw}")
        if (not (node in alreadyDraw)) and (not (node in isolatedNodes)):
            iterator += 1
            plt.figure(figsize=(28, 28))
            # plt.title(f"Connection net, window: " +str(windowsize) + " step: "+str(step) + " Part " + str(iterator))
            # plt.title(f"Connection net: " + " Part " + str(iterator)+f"\n\nPir/Pir0 >= 0.5\nAlpha = {alpha}", fontdict={'fontsize': 24})
            plt.axis("off")
            plt.tight_layout()
            # plt.text(x=0, y=20, s=f"Pir/Pir0 >= 0.5\nAlpha = {alpha}")
            OneGraph = nx.ego_graph(GUFE, node, radius=None)
            primary_edges = []
            green_edges = []
            blue_edges = []
            red_edges = []
            # secondary nodes recoloring (try not only for 3 but in cycle for all)
            for first_nodes in OneGraph:
                primary_edges.append(OneGraph.edges(first_nodes))
            edges_len = [len(list(edges)) for edges in primary_edges]
            if max(edges_len) > 4:
                red_edges = primary_edges[np.argmax(edges_len)]
                banned_nodes = [list(OneGraph.nodes)[np.argmax(edges_len)]]
                # print(banned_nodes)
                for node1, node2 in red_edges:
                    secondary_edges = (OneGraph.edges(node2))
                    right_secondary_edges = [edge for edge in secondary_edges if edge[1] not in banned_nodes]
                    green_edges.extend(right_secondary_edges)
                    banned_nodes.append(node2)
                for node1, node2 in green_edges:
                    tertiary_edges = (OneGraph.edges(node2))
                    right_tertiary_edges = [edge for edge in tertiary_edges if edge[1] not in banned_nodes]
                    blue_edges.extend(right_tertiary_edges)
                    banned_nodes.append(node2)

            # print(green_edges)
            # print(blue_edges)
            newlabels = {key: labelsd[key] for key in list(OneGraph.nodes)}
            # newEdgelabels = dict([((n1, n2), f'E = {UFE[n1, n2]} LD = {ld[n1, n2]} UFE0 = {UFE0[n1,n2]}') for n1, n2 in OneGraph.edges])
            newEdgelabels = dict(
                [((n1, n2), f'E = {UFE[n1, n2]}') for n1, n2 in
                 OneGraph.edges])
            # print(newEdgelabels)
            # plt.text(1, 1, "runs " + runs + "\nEpi " + epi + "\nL " + seqlen + "\nr " + recomb)
            nx.draw_networkx(GUFE, pos=nx.circular_layout(OneGraph, scale=4), with_labels=True,
                             labels=newlabels, nodelist=OneGraph.nodes, edgelist=OneGraph.edges,
                             **options)  # Maybe just draw Ego?
            # edge color drawing
            nx.draw_networkx_edges(GUFE, pos=nx.circular_layout(OneGraph, scale=4), edgelist=red_edges,
                                   edge_color="red", width=4)
            nx.draw_networkx_edges(GUFE, pos=nx.circular_layout(OneGraph, scale=4), edgelist=green_edges,
                                   edge_color="green", width=3)
            nx.draw_networkx_edges(GUFE, pos=nx.circular_layout(OneGraph, scale=4), edgelist=blue_edges,
                                   edge_color="blue", width=2)
            nx.draw_networkx_edge_labels(GUFE, nx.circular_layout(OneGraph, scale=4), nodelist=OneGraph.nodes,
                                         edge_labels=newEdgelabels, font_size=80)

            plt.savefig("YourDataPath/" + dataname + " Part " + str(iterator) + ".png")
            # plt.show()
            plt.close()
            alreadyDraw.extend(list(OneGraph))
            alledges = [
                f"all\t{labelsd[item[0]]}\t{labelsd[item[0]]}\tall\t{labelsd[item[1]]}\t{labelsd[item[1]]}  color=black,thickness=5p\n"
                for item in OneGraph.edges]
            rededges = [
                f"all\t{labelsd[item[0]]}\t{labelsd[item[0]]}\tall\t{labelsd[item[1]]}\t{labelsd[item[1]]}  color=red_a3,thickness=5p\n"
                for item in red_edges]
            blueedges = [
                f"all\t{labelsd[item[0]]}\t{labelsd[item[0]]}\tall\t{labelsd[item[1]]}\t{labelsd[item[1]]}  color=blue_a3,thickness=5p\n"
                for item in blue_edges]
            greenedges = [
                f"all\t{labelsd[item[0]]}\t{labelsd[item[0]]}\tall\t{labelsd[item[1]]}\t{labelsd[item[1]]}  color=green_a3,thickness=5p\n"
                for item in green_edges]
            # with open("RealDataNewFigsFirstRealignedBigTest/" + dataname + " Part "+str(iterator)+"all_edges"+".txt", "w") as output:
            #    output.writelines(alledges)
            # with open("RealDataNewFigsFirstRealignedBigTest/" + dataname + " Part "+str(iterator)+"red_edges"+".txt", "w") as output:
            #     output.writelines(rededges)
            # with open("RealDataNewFigsFirstRealignedBigTest/" + dataname + " Part "+str(iterator)+"green_edges"+".txt", "w") as output:
            #    output.writelines(greenedges)
            # with open("RealDataNewFigsFirstRealignedBigTest/" + dataname + " Part "+str(iterator)+"blue_edges"+".txt", "w") as output:
            #    output.writelines(blueedges)
            with open("YourDataPath" + dataname + " Part " + str(iterator) + "every_edge" + ".txt",
                      "w") as output:
                output.writelines(alledges + rededges + greenedges + blueedges)


def find_uefi(array: np.ndarray, fcut=0.05, template=None) -> np.ndarray:
    if template is None:
        template = np.ones((array.shape[0],array.shape[0]))
        #print(template)
    length = range(array.shape[0])
    # print(length[39])
    length2 = range(1, array.shape[0])
    # print(length2)
    width = array.shape[1]
    # print(width)
    uefi = np.zeros((array.shape[0], array.shape[0]))

    for i in length:
        for j in length2[i:]:
            if template[i, j]:
                # print(f"current pos{i},{j}")
                f = {"00": 0, "01": 0, "10": 0, "11": 0}
                for k in range(width):
                    f[str(array[i, k]) + str(array[j, k])] += 1
                f = {key: value / width for (key, value) in f.items()}
                # print(f)
                # print(min(f.values()) > fcut)
                if min(f.values()) > fcut and f["01"] * f["10"] != f["00"] ** 2:
                    #print("one-one")
                    uefi[i, j] = round(1 - (
                            (math.log(f["11"] / f["00"])) / (math.log((f["01"] * f["10"]) / (f["00"] ** 2)))), 4)
                    uefi[j, i] = round(1 - (
                            (math.log(f["11"] / f["00"])) / (math.log((f["01"] * f["10"]) / (f["00"] ** 2)))), 4)
           # else:
              #  print(template[i, j])
    return uefi


def ufe(array: np.ndarray, i: int, j: int, fcut=0) -> float:
    width = array.shape[1]
    f = {"00": 0, "01": 0, "10": 0, "11": 0}
    for k in range(width):
        f[str(array[i, k]) + str(array[j, k])] += 1
    f = {key: value / width for (key, value) in f.items()}
    if min(f.values()) > fcut and f["01"] * f["10"] != f["00"] ** 2:
       # if i == 5 and j == 16:
           # print(1 - ((math.log(f["11"] / f["00"])) / (math.log((f["01"] * f["10"]) / (f["00"] ** 2)))))

        return (1 - ((math.log(f["11"] / f["00"])) / (math.log((f["01"] * f["10"]) / (f["00"] ** 2)))))
    else:
        #print(f"Strange{min(f.values())}")
        return None


def set_link_zeroes(array: np.ndarray, pos: int) -> np.ndarray:  # можно попробовать уменьшить размер передаваемого массива
    cleared_array = array[:, array[pos, :] < 1]
    return cleared_array


def score_diff(array: np.ndarray, ref_array: np.ndarray):
    false_pos = 0
    not_found = 0
    links = np.sum(np.sum(ref_array)) / 2
    score = array - ref_array
    for i in range(array.shape[0]):
        for j in range(array.shape[0])[i:]:

            if score[i, j] > 0:
                false_pos += 1
            if score[i, j] < 0:
                not_found += 1

    return [false_pos / links, not_found / links]


def find_ufei0(array: np.ndarray, link_matrix: np.ndarray, fcut=0):
    length = link_matrix.shape[0]
    ufe0 = np.zeros(link_matrix.shape)
    link_graph = nx.from_numpy_array(link_matrix)
    for i in range(length - 1):
        links_i = set(nx.ego_graph(link_graph,i,radius=1,center=False).nodes)
        # print(links_i)
        for j in range(i + 1, length):
            # print(i, j)
            zeroflag = True
            if link_matrix[i, j]:
                # print(i, j)
                links_j = set(nx.ego_graph(link_graph,j,radius=1,center=False).nodes)
                # print(links_j)
                links_j = links_j - links_i
                # print(links_i)
                # print(links_j)
                if links_j or links_i:
                    # print("Hurray!")
                    all_ufe0 = []
                    if links_j:
                        for link in links_j:
                            if link != i:
                                # print("not hehe")
                               #if i == 5 and j == 16:
                                   # print(link)
                                curr_ufe = ufe(set_link_zeroes(array, link), i, j, fcut)
                                if curr_ufe is None:
                                    #print("Hmm")
                                    zeroflag = False
                                    ufe0[i, j] = 0
                                    ufe0[j, i] = 0
                                    break
                                all_ufe0.append(curr_ufe)
                    if links_i:
                        for link in links_i:
                            # print("hehe")
                            if link != j:
                               # if i == 5 and j == 16:
                                    #print(link)
                                curr_ufe = ufe(set_link_zeroes(array, link), i, j, fcut)
                                if curr_ufe is None:
                                    #print("Hmm")
                                    zeroflag = False
                                    ufe0[i, j] = 0
                                    ufe0[j, i] = 0
                                    break
                                all_ufe0.append(curr_ufe)
                    # print(all_ufe0)
                    if zeroflag:
                        ufe0[i, j] = round(min(all_ufe0),4)
                        ufe0[j, i] = round(min(all_ufe0),4)
            # print(ufe0[i, j])

    return ufe0


if __name__ == '__main__':
    # Data setup
    datavalpath = "YourDataNameVal"
    datacolpath = "YourDataNameCol"

    arrayFull = np.transpose(np.load(f"{datavalpath}.npy", allow_pickle=True))
    labels = np.load(f"{datacolpath}.npy")

    """
    if using mock sequneces (Don`t forget to comment Data setup for real data)
    # Data setup
    array_raw = loadmat("YourDataName".mat")["dat"]
    array_new = array_raw[0, 0]
    for a in range(array_raw.shape[1] - 1):
        array_new = np.concatenate((array_new, array_raw[0, a + 1]), 0)
    arrayFull = np.transpose(array_new)
    labels = list(range(0,arrayFull[0]))
    """

    sumarray = np.sum(arrayFull, axis=0)
    mask1 = np.multiply(0 <= sumarray, sumarray <= 20)
    mask2 = np.multiply(20 < sumarray, sumarray <= 50)

    # GPU Setup
    data_length = arrayFull.shape[0]
    threads_per_block = 32

    # Functions call (GPU)

    arraysub = array_subdivV3(arrayFull, [0, 20, 50], [sum(mask1), sum(mask2)])  # Change to appropriate values
    data_length = arraysub.shape[0]
    (UFE_Mask, UFE, UFE0) = find_links_UFE_GPU(arraysub)
    draw_connections_cluster(labels,UFE_Mask,UFE)


"""
Functions call (Non GPU) (Not recommended)
    UFE2 = find_uefi(arraysub,0)
    ufe_mask2 = UFE2 > 0
    UFE02 = find_ufei0(arraysub,ufe_mask2)
    UFE0div2 = np.zeros(UFE.shape)
    for i2 in range(UFE0div2.shape[0]):
        for j2 in range(UFE0div2.shape[1]):
            if UFE2[i2, j2] != 0:
                UFE0div2[i2, j2] = (UFE02[i2, j2] / UFE2[i2, j2])
    UFEwZeroCut2 = UFE0div2 >= 0.5
    draw_connections_cluster(labels, UFEwZeroCut2, UFE2)
"""