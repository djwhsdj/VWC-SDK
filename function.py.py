import math

def utilization(pw_r, pw_c, k_r, k_c, array_r, array_c, inc, ouc, arcycle, accycle) :
  
  total_mcells = array_r * array_c * arcycle * accycle
  used_col = ouc * (pw_r - k_r + 1) * (pw_c - k_c + 1) / (array_c * accycle)
  used_t_mcells = total_mcells * used_col
  used_mcells = (pw_r * pw_c * inc)/(array_r * arcycle) * ((k_r * k_c)/(pw_r * pw_c))
  util = used_mcells * used_t_mcells / total_mcells
  util = round(util, 3)
  return util

def im2col (image_col, image_row, filter_col, filter_row, in_channel, out_channel, array_row, array_col, \
                Ks, memory_precision, bit_precision) :

    pw_row = filter_row + Ks - 1
    pw_col = filter_col + Ks - 1
    PWs_w = pw_row - filter_row + Ks
    PWs_h = pw_col - filter_col + Ks

    row_slide = math.ceil((image_row - pw_row)/PWs_w) + 1
    col_slide = math.ceil((image_col - pw_col)/PWs_h) + 1

    real_array_col = math.floor(array_col * memory_precision / bit_precision)

    col_cycle = math.ceil(out_channel/real_array_col)
    row_cycle = math.ceil(filter_row*filter_col*in_channel/array_row)
    total_cycle = col_slide * row_slide * row_cycle * col_cycle
    N_PW = col_slide * row_slide
    N_sub = row_cycle * col_cycle
    N_row = filter_row * filter_col * in_channel
    N_col = out_channel * bit_precision
    
    return total_cycle, N_PW, N_sub, N_row, N_col, row_cycle, col_cycle


def SDK (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col, Ks, memory_precision, bit_precision) :
    
    row_vector = filter_row * filter_col * in_channel
    col_vector = out_channel
    
    real_array_col = math.floor(array_col * memory_precision / bit_precision)
    used_row = math.ceil(row_vector/array_row)
    used_col = math.ceil(col_vector/real_array_col)
    
    new_array_row = array_row * used_row
    new_array_col = array_col * used_col

    # initialize
    cycle = []
    N_PW = []
    N_sub = []
    N_row = []
    N_col = []
    w = []
    w.append(filter_row*filter_col)
    cycle.append(used_row*used_col*(image_row-filter_row+1)*(image_col-filter_col+1))
    N_PW.append((image_row-filter_row+1)*(image_col-filter_col+1))
    N_sub.append(used_row*used_col)
    N_row.append(filter_row*filter_col*in_channel)
    N_col.append(out_channel*bit_precision)
    
    i=0

    while True :
        i += 1
        pw_row = filter_row + i + Ks - 2
        pw_col = filter_col + i + Ks - 2
        PWs_w = pw_row - filter_row + Ks
        PWs_h = pw_col - filter_col + Ks
        pw = pw_row * pw_col
        if pw*in_channel <= new_array_row and i * i * out_channel <= new_array_col :
            parallel_window_row = math.ceil((image_row - pw_row)/PWs_w) + 1
            parallel_window_col = math.ceil((image_col - pw_col)/PWs_h) + 1
            
            if parallel_window_row * parallel_window_row * used_row * used_col <= cycle[0] :
                del cycle[0]
                del w[0]
                del N_PW[0]
                del N_sub[0]
                del N_row[0]
                del N_col[0]

                cycle.append(parallel_window_row * parallel_window_col * used_row * used_col)
                w.append(pw)
                N_PW.append(parallel_window_row * parallel_window_col)
                N_row.append(pw*in_channel)
                N_col.append((pw_row-filter_row+1)*(pw_col-filter_col+1)*out_channel*bit_precision)
                N_sub.append(used_row * used_col)
            
        else :
            break
        
    return cycle[0], N_PW[0], N_sub[0], N_row[0], N_col[0], w[0], used_row, used_col

def vw_sdk (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col, Ks, memory_precision, bit_precision) :
    
    real_used_col = math.floor(array_col * memory_precision / bit_precision)

    i = 0 # initialize # overlap col
    j = 1 # overlap row

    reg_total_cycle = [] # initialize
    reg_overlap_row = []
    reg_overlap_col = []
    reg_row_cycle = []
    reg_col_cycle = []
    reg_ICt = []
    reg_OCt = []
    reg_n_PW = []
    reg_row = []
    reg_col = []
    cnt = 1
    while True :
        try :
            i += 1
            if i + filter_col - 1 > image_col : 
                i = 1
                j += 1
                cnt += 1
                if j + filter_row - 1 > image_row : 
                    break
            
            pw_row = filter_row + i + Ks - 2
            pw_col = filter_col + j + Ks - 2
            PWs_w = pw_row - filter_row + Ks
            PWs_h = pw_col - filter_col + Ks

            reg_N_parallel_window_row = math.ceil((image_row - pw_row)/PWs_w) + 1
            reg_N_parallel_window_col = math.ceil((image_col - pw_col)/PWs_h) + 1

            
            # for cycle computing
            # Tiled IC
            if in_channel == 3 :
                ICt = math.floor(array_row /(filter_row+i-1)*(filter_col+j-1))
                if ICt > in_channel :
                    ICt = 3
                row_cycle = math.ceil(in_channel / ICt)
            else :
                ICt = math.floor(array_row /(pw_row*pw_col))
                row_cycle = math.ceil(in_channel / ICt)
            
            # Tiled OC
            OCt =  math.floor(real_used_col / (i * j))
            col_cycle = math.ceil(out_channel / OCt)
    
            reg_N_of_computing_cycle = reg_N_parallel_window_row * reg_N_parallel_window_col \
                                    * row_cycle * col_cycle
            
            if i == 1 : # initialize
                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_n_PW.append(reg_N_parallel_window_row * reg_N_parallel_window_col)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)
                reg_row.append(pw_row*pw_col*in_channel)
                reg_col.append(out_channel*bit_precision)

            if reg_total_cycle[0] > reg_N_of_computing_cycle :
                del reg_total_cycle[0]
                del reg_n_PW[0]
                del reg_overlap_row[0]
                del reg_overlap_col[0]
                del reg_row_cycle[0]
                del reg_col_cycle[0]
                del reg_ICt[0]
                del reg_OCt[0]
                del reg_row[0]
                del reg_col[0]

                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_n_PW.append(reg_N_parallel_window_row * reg_N_parallel_window_col)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)
                reg_row.append(pw_row*pw_col*in_channel)
                reg_col.append((pw_row-filter_row+1)*(pw_col-filter_col+1)*out_channel*bit_precision)

    
        except ZeroDivisionError :
            continue
            
    # print(reg_overlap_row, reg_overlap_col, reg_total_cycle[0])
    return reg_total_cycle[0], reg_n_PW[0], reg_row_cycle[0]*reg_col_cycle[0], reg_row[0], reg_col[0], reg_overlap_col[0], reg_overlap_row[0], reg_ICt[0], reg_OCt[0], reg_row_cycle[0], reg_col_cycle[0]

def result (network, image, array, kernel, channel, stride, M_bit, W_bit) :
    # im2col
    CC_im2col, PW_im2col, sub_im2col, row_im2col, col_im2col, ar_im2col, ac_im2col = [], [], [], [], [], [], []

    # SDK
    CC_SDK, PW_SDK, sub_SDK, row_SDK, col_SDK, PW_size, ar_SDK, ac_SDK = [], [], [], [], [], [], [], []

    #
    CC_VWSDK, PW_VWSDK, sub_VWSDK, row_VWSDK, col_VWSDK, height_VWSDK, width_VWSDK, ar_VWSDK, ac_VWSDK = [], [], [], [], [], [], [], [], []
    IC_tiled, OC_tiled= [], []

    
    print("="*50)
    print(" RESULTS of COMPUTING CYCLES")
    print("-"*30)

    if network == 'CNN8' :
      for i in range(len(image)) :
          T_im2col, I_PW, I_sub, I_row, I_col, ar_i, ac_i = im2col(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i+1], array[0], array[1], stride[i], M_bit, W_bit)
          CC_im2col.append(T_im2col)
          PW_im2col.append(I_PW)
          sub_im2col.append(I_sub)
          row_im2col.append(I_row)
          col_im2col.append(I_col)
          ar_im2col.append(ar_i)
          ac_im2col.append(ac_i)

          T_SDK, S_PW, S_sub, S_row, S_col, S_w, ar_s, ac_s = SDK(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i+1], array[0], array[1], stride[i], M_bit, W_bit)
          CC_SDK.append(T_SDK)
          PW_SDK.append(S_PW)
          sub_SDK.append(S_sub)
          row_SDK.append(S_row)
          col_SDK.append(S_col)
          PW_size.append(S_w)
          ar_SDK.append(ar_s)
          ac_SDK.append(ac_s)

          T_cycle, V_PW, V_sub, V_row, V_col, SDK_h, SDK_w, tiled_IC, tiled_OC, ar_v, ac_v = vw_sdk(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i+1], array[0], array[1], stride[i], M_bit, W_bit)
          CC_VWSDK.append(T_cycle)
          PW_VWSDK.append(V_PW)
          sub_VWSDK.append(V_sub)
          row_VWSDK.append(V_row)
          col_VWSDK.append(V_col)
          height_VWSDK.append(SDK_h)
          width_VWSDK.append(SDK_w)
          IC_tiled.append(tiled_IC)
          OC_tiled.append(tiled_OC)
          ar_VWSDK.append(ar_v)
          ac_VWSDK.append(ac_v)

      for i in range(len(image)) :
        # def utilization(pw_r, pw_c, k_r, k_c, array_r, array_c, inc, ouc, arcycle, accycle) :
        print(" CONV LAYER "+ str(i+1))
        print("    Im2col = {}".format(CC_im2col[i]))
        print("      - shape of PW = {} x {} x {} x {}".format(kernel[i], kernel[i], channel[i], channel[i+1]))
        print("      - # of rows = {}".format(row_im2col[i]))
        print("      - # of cols = {}".format(col_im2col[i]))
        print("      - # of PWs  = {}".format(PW_im2col[i]))
        print("      - # of sub-arrays = {}".format(sub_im2col[i]))
        print("      - utilization = {}".format(utilization(kernel[i], kernel[i],kernel[i], kernel[i],array[0], array[1],channel[i], channel[i+1], ar_im2col[i], ac_im2col[i])))
        
        print("    SDK    = {}".format(CC_SDK[i]))
        print("      - shape of PW = {} x {} x {} x {}".format(int(math.sqrt(PW_size[i])), int(math.sqrt(PW_size[i])), channel[i], channel[i+1]))
        print("      - # of rows = {}".format(row_SDK[i]))
        print("      - # of cols = {}".format(col_SDK[i]))
        print("      - # of PWs  = {}".format(PW_SDK[i]))
        print("      - # of sub-arrays = {}".format(sub_SDK[i]))
        print("      - utilization = {}".format(utilization(int(math.sqrt(PW_size[i])), int(math.sqrt(PW_size[i])), kernel[i], kernel[i],array[0], array[1],channel[i], channel[i+1], ar_SDK[i], ac_SDK[i])))
        
        
        if CC_VWSDK[i] >= CC_im2col[i] :
          CC_VWSDK[i] = CC_im2col[i]
          print("    VW-SDK = {}".format(CC_VWSDK[i]))
          print("      - Optimal shape of PW = {} x {} x {} x {}".format(kernel[i], kernel[i], channel[i], channel[i+1]))
          print("      - # of rows = {}".format(row_im2col[i]))
          print("      - # of cols = {}".format(col_im2col[i]))
          print("      - # of PWs  = {}".format(PW_im2col[i]))
          print("      - # of sub-arrays = {}".format(sub_im2col[i]))
          print("      - Reduction Compared to Im2col = {:.2f} %".format((CC_im2col[i]-CC_VWSDK[i])/CC_im2col[i]*100))
          print("      - Reduction Compared to SDK    = {:.2f} %".format((CC_SDK[i]-CC_VWSDK[i])/CC_SDK[i]*100))    
          print("      - utilization = {}".format(utilization(kernel[i], kernel[i],kernel[i], kernel[i],array[0], array[1],channel[i], channel[i+1], ar_im2col[i], ac_im2col[i])))
        
        else :
          print("    VW-SDK = {}".format(CC_VWSDK[i]))
          print("      - Optimal shape of PW = {} x {} x {} x {}".format(kernel[i] + width_VWSDK[i]-1, kernel[i] + height_VWSDK[i]-1, IC_tiled[i], OC_tiled[i]))
          print("      - # of rows = {}".format((row_VWSDK[i])))
          print("      - # of col = {}".format((col_VWSDK[i])))
          print("      - # of PWs  = {}".format(PW_VWSDK[i]))
          print("      - # of sub-arrays = {}".format(sub_VWSDK[i]))
          print("      - Reduction Compared to Im2col = {:.2f} %".format((CC_im2col[i]-CC_VWSDK[i])/CC_im2col[i]*100))
          print("      - Reduction Compared to SDK    = {:.2f} %".format((CC_SDK[i]-CC_VWSDK[i])/CC_SDK[i]*100))   
          print("      - utilization = {}".format(utilization(kernel[i] + width_VWSDK[i]-1, kernel[i] + height_VWSDK[i]-1,kernel[i], kernel[i],array[0], array[1],channel[i], channel[i+1], ar_VWSDK[i], ac_VWSDK[i])))
         
      print("="*50)
    
    elif network == 'Resnet20' :

      for i in range(len(image)) :
        if i == 0 :
          T_im2col, I_PW, I_sub, I_row, I_col, ar_i, ac_i = im2col(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i+1], array[0], array[1], stride[i], M_bit, W_bit)
          T_SDK, S_PW, S_sub, S_row, S_col, S_w, ar_s, ac_s = SDK(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i+1], array[0], array[1], stride[i], M_bit, W_bit)
          T_cycle, V_PW, V_sub, V_row, V_col, SDK_h, SDK_w, tiled_IC, tiled_OC, ar_v, ac_v = vw_sdk(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i+1], array[0], array[1], stride[i], M_bit, W_bit)
        
        else :
          T_im2col, I_PW, I_sub, I_row, I_col, ar_i, ac_i = im2col(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i], array[0], array[1], stride[i], M_bit, W_bit)
          T_SDK, S_PW, S_sub, S_row, S_col, S_w, ar_s, ac_s = SDK(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i], array[0], array[1], stride[i], M_bit, W_bit)
          T_cycle, V_PW, V_sub, V_row, V_col, SDK_h, SDK_w, tiled_IC, tiled_OC, ar_v, ac_v = vw_sdk(image[i], image[i], kernel[i], kernel[i], channel[i], channel[i], array[0], array[1],stride[i], M_bit, W_bit)
          
        CC_im2col.append(T_im2col)
        PW_im2col.append(I_PW)
        sub_im2col.append(I_sub)
        row_im2col.append(I_row)
        col_im2col.append(I_col)
        ar_im2col.append(ar_i)
        ac_im2col.append(ac_i)
        
        CC_SDK.append(T_SDK)
        PW_SDK.append(S_PW)
        sub_SDK.append(S_sub)
        row_SDK.append(S_row)
        col_SDK.append(S_col)
        PW_size.append(S_w)   
        ar_SDK.append(ar_s)
        ac_SDK.append(ac_s)  
        
        CC_VWSDK.append(T_cycle)
        PW_VWSDK.append(V_PW)
        sub_VWSDK.append(V_sub)
        row_VWSDK.append(V_row)
        col_VWSDK.append(V_col)
        height_VWSDK.append(SDK_h)
        width_VWSDK.append(SDK_w)
        IC_tiled.append(tiled_IC)
        OC_tiled.append(tiled_OC)
        ar_VWSDK.append(ar_v)
        ac_VWSDK.append(ac_v)

      for i in range(len(image)) :
        print(" CONV LAYER "+ str(i+1))
        print("    Im2col = {}".format(CC_im2col[i]))
        print("      - # of rows = {}".format(row_im2col[i]))
        print("      - # of cols = {}".format(col_im2col[i]))
        print("      - # of PWs  = {}".format(PW_im2col[i]))
        print("      - # of sub-arrays = {}".format(sub_im2col[i]))
        print("      - utilization = {}".format(utilization(kernel[i], kernel[i], kernel[i], kernel[i],array[0], array[1],channel[i], channel[i], ar_im2col[i], ac_im2col[i])))
        

        print("    SDK    = {}".format(CC_SDK[i]))
        print("      - shape of PW = {} x {} x {} x {}".format(int(math.sqrt(PW_size[i])), int(math.sqrt(PW_size[i])), channel[i], channel[i]))
        print("      - # of rows = {}".format(row_SDK[i]))
        print("      - # of cols = {}".format(col_SDK[i]))
        print("      - # of PWs  = {}".format(PW_SDK[i]))
        print("      - # of sub-arrays = {}".format(sub_SDK[i]))
        print("      - utilization = {}".format(utilization(int(math.sqrt(PW_size[i])), int(math.sqrt(PW_size[i])), kernel[i], kernel[i],array[0], array[1],channel[i], channel[i], ar_im2col[i], ac_im2col[i])))
        

        if CC_VWSDK[i] > CC_im2col[i] :
          CC_VWSDK[i] = CC_im2col[i]
          print("    VW-SDK = {}".format(CC_VWSDK[i]))
          if i == 0 :
            print("      - Optimal shape of PW = {} x {} x {} x {}".format(kernel[i], kernel[i], channel[i], channel[i+1]))
          else :
            print("      - Optimal shape of PW = {} x {} x {} x {}".format(kernel[i], kernel[i], channel[i], channel[i]))
          print("      - # of rows = {}".format(row_im2col[i]))
          print("      - # of cols = {}".format(col_im2col[i]))
          print("      - # of PWs  = {}".format(PW_im2col[i]))
          print("      - # of sub-arrays = {}".format(sub_im2col[i]))
          print("      - Reduction Compared to Im2col = {:.2f} %".format((CC_im2col[i]-CC_VWSDK[i])/CC_im2col[i]*100))
          print("      - Reduction Compared to SDK    = {:.2f} %".format((CC_SDK[i]-CC_VWSDK[i])/CC_SDK[i]*100))    
          print("      - utilization = {}".format(utilization(kernel[i], kernel[i],kernel[i], kernel[i],array[0], array[1],channel[i], channel[i], ar_im2col[i], ac_im2col[i])))
        

        else :
          print("    VW-SDK = {}".format(CC_VWSDK[i]))
          print("      - Optimal shape of PW = {} x {} x {} x {}".format(kernel[i] + width_VWSDK[i]-1, kernel[i] + height_VWSDK[i]-1, IC_tiled[i], OC_tiled[i]))
          print("      - # of rows = {}".format((row_VWSDK[i])))
          print("      - # of cols = {}".format(col_VWSDK[i]))
          print("      - # of PWs  = {}".format(PW_VWSDK[i]))
          print("      - # of sub-arrays = {}".format(sub_VWSDK[i]))
          print("      - Reduction Compared to Im2col = {:.2f} %".format((CC_im2col[i]-CC_VWSDK[i])/CC_im2col[i]*100))
          print("      - Reduction Compared to SDK    = {:.2f} %".format((CC_SDK[i]-CC_VWSDK[i])/CC_SDK[i]*100))  
          print("      - utilization = {}".format(utilization(kernel[i] + width_VWSDK[i]-1, kernel[i] + height_VWSDK[i]-1,kernel[i], kernel[i],array[0], array[1],channel[i], channel[i], ar_VWSDK[i], ac_VWSDK[i])))
           
      print("="*50)