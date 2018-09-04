import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, y, sl,last = [], [], [], [],[]
    for t in ts:
      u.append(t[0])
      i.append([t[2]]+t[3])
      sub_sample_size=len(t[3])+1
      mask=np.zeros(sub_sample_size,np.int64)
      mask[0]=1
      y.append(mask)
      sl.append(len(t[1]))#histroy click seq
      last.append(t[1][-1])#last click
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)
    #print('u',u)
    #print('i',len(i))
    #print('y',y)
    #print('hist_i',hist_i)
    #print('sl',sl)
    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1
    #print('hist_i',hist_i)
    return self.i, (u, i, y, hist_i, sl,last)



class DataInputTest:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, sl,last = [], [],[]
    for t in ts:
      u.append(t[0])
     
      sl.append(len(t[1]))#histroy click seq
      last.append(t[1][-1])#last click
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)
    #print('u',u)
    #print('i',len(i))
    #print('y',y)
    #print('hist_i',hist_i)
    #print('sl',sl)
    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1
    #print('hist_i',hist_i)
    return self.i, (u, hist_i, sl,last)



class DataInputEval:
    def __init__(self, data,item_map, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.item_map=item_map
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, sl, last = [], [], []
        for t in ts:
            u.append(t[0])

            sl.append(len(t[1]))  # histroy click seq
            last.append(t[1][-1])  # last click
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        # print('u',u)
        # print('i',len(i))
        # print('y',y)
        # print('hist_i',hist_i)
        # print('sl',sl)
        k = 0
        for t in ts:
   
            for l in range(len(t[1])):
                t_idx=self.item_map[t[1][l]]
                hist_i[k][l] = t_idx

            last[k]=self.item_map[last[k]]
            k+=1
        # print('hist_i',hist_i)
        return self.i, (u, hist_i, sl, last)
