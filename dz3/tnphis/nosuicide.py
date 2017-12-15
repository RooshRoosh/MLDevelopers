import os
import multiprocessing
import datetime as dt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import chi2, SelectPercentile, mutual_info_classif, SelectFromModel
from scipy.sparse import lil_matrix, vstack, hstack
import pymorphy2
from sklearn.decomposition import TruncatedSVD

def write_data(data):
  with open(os.path.abspath(os.path.dirname(__file__) + './out.csv'), 'w') as outf:
    outf.write('id,prob\n')
    for itm in data:
      outf.write(str(int(itm[0])) + ',' + str(float(itm[1])) + '\n')
    outf.close()

def normalize_text(txt):
  myan = pymorphy2.MorphAnalyzer()
  newtxt = ''
  for w in txt.decode('utf8').split(' '):
    myword = myan.parse(w.lower())
    newtxt += myword[0].normal_form

  return newtxt.encode('utf8')

def normalize_array(data):
  for idx, txt in enumerate(data):
    data[idx] = normalize_text(txt)
  return data

def geld_data(X, Y, vectorizer, Classifier):
  # culling?
  #return (X, Y)
  print('WTF2', X.shape, Y.shape)
  myTmpClassifier = Classifier(X = X, Y = Y, test_data = X, vectorizer = vectorizer)
  new_X_train = None
  new_Y_train = []
  threshold = 1 - 1 / X.shape[1] ** 2
  my_full_estimates = myTmpClassifier.run_crosscheck()
  my_estimates = my_full_estimates['proba']
  for idx, row in enumerate(my_estimates):
    if abs(row - Y[idx]) <= threshold:
      if new_X_train is None:
        new_X_train = X[idx]
      else:
        new_X_train = vstack([new_X_train, X[idx]])
      new_Y_train.append(Y[idx])
    else:
      pass
      #print(row)
      #print(td[1][idx])
  new_Y_train = np.array(new_Y_train)

  print('SHAPE', new_X_train.shape)

  return (new_X_train, new_Y_train)

def load_text_data(total_pts, labels = True, **args):
  with open(os.path.abspath(os.path.dirname(__file__) + './' + args['filename']), 'r') as td:
    index = -1
    X = np.zeros(
      (total_pts),
      dtype = [('id', 'u2'), ('text', 'S16000')]
    )
    if labels:
      Y = np.zeros(total_pts)

    for line in td:
      sl = line.split(',')
      if index > -1 and index < total_pts:
        X[index] = (sl[0], sl[1].encode('utf8'))
        if labels:
          Y[index] = sl[2]

      index += 1

    td.close()

  if labels:
    return (X, Y)
  else:
    return X

def vectorize_data(**args):
  total_data_pts = len(args['td'])
  all_docs = np.zeros(
      total_data_pts * 2,
      dtype = [('id', 'u2'), ('text', 'S16000')]
  )

  all_docs[:total_data_pts] = args['td']
  all_docs[total_data_pts:] = args['testd']

  myvec = CustomTextVectorizer(
    data = all_docs,
    stop_words = args.get('stop_words'),
    vectorizer = args.get('vectorizer'),
  )

  current_vectorizer = myvec.vectorize(args['td']['text'])
  tv = myvec.dump()

  myvec.vectorize(args['testd']['text'])
  testv = myvec.dump()

  return (tv, testv, current_vectorizer)

class CustomTextVectorizer():
  def __init__(self, **args):
    self.data = args['data']
    self.labels = args.get('labels')
    self.stop_words = args.get('stop_words')
    self.fn = args.get('fn')
    self.vecs = None

    if self.fn is None:
      self.fn = 'training_vecs.csv'

    # best so far
    # stop_words = self.stop_words,
    # ngram_range = (2, 3),
    # max_df = .2,
    # max_features = 16000

    # after normalization & new stop words:
    # stop_words = self.stop_words,
    # ngram_range = (2, 3),
    # max_df = .7,
    # max_features = 128000
    # ~.5 precision & recall on test data

    # works with culling
    # analyzer = 'char',
    # stop_words = self.stop_words,
    # ngram_range = (3, 4),
    # max_df = .7,
    # max_features = 6000
    # 48k features best on train set.
    if args.get('vectorizer') is not None:
      self.vectorizer = args.get('vectorizer')
    else:
      self.vectorizer = TfidfVectorizer(
        stop_words = self.stop_words,
        ngram_range = (2, 3),
        max_df = .9,
        max_features = 16000
      )

    self.vectorizer.fit(self.data['text'])
    print(len(self.vectorizer.vocabulary_.items()))

  def vectorize(self, data):
    self.vecs = self.vectorizer.transform(data)
    return self.vectorizer

  def write(self):
    with open(os.path.abspath(os.path.dirname(__file__) + './' + self.fn), 'w') as outf:
      outf.write('id,' + ','.join(['f' + str(i) for i in range(len(self.vecs.toarray()[0]))]) + ',label\n')
      for index, itm in enumerate(self.vecs.toarray()):
        current_row = str(self.data['id'][index]) + ',' + ','.join(list(str(f) for f in itm))
        if self.labels is not None:
          current_row += ',' + str(int(self.labels[index]))
        outf.write(current_row + '\n')
      outf.close()

  def dump(self):
    return self.vecs

class StrangeClassifier():
  def __init__(self, **args):
    self.ids = args.get('ids')
    self.X = args['X']
    self.Y = args['Y']
    self.test_data = args['test_data']
    print('EFFIN_TEST_DATA', self.test_data.shape)
    self.total_data_pts = self.test_data.shape[0]
    self.estimates = np.zeros((self.total_data_pts, 2))
    self.crosscheck_estimates = None
    self.calc_start = dt.datetime.now()
    self.vectorizer = args['vectorizer']

  def estimate(self):
    # class_prior = [.9, .1] - we dunnno
    classifier = BernoulliNB()
    #classifier = MultinomialNB(alpha = 0.02)
    #classifier = DecisionTreeClassifier(class_weight = { 0: 1, 1: 9 })
    #classifier = KNeighborsClassifier(n_neighbors=50, metric='minkowski', p=3)
    # classifier = RandomForestClassifier(
    #   max_depth = 32,
    #   n_estimators = 64,
    #   max_features = 0.25,
    #   class_weight = { 0: 1, 1: 9 },
    #   n_jobs = 3
    # )
    classifier.fit(self.X, self.Y)
    if self.calc_start is not None:
      print('Fitting time: ' + str((dt.datetime.now() - self.calc_start).total_seconds()) + 's')

    #if self.ids is not None:
    results_proba = classifier.predict_proba(self.test_data)
    if self.calc_start is not None:
      print('Prediction time: ' + str((dt.datetime.now() - self.calc_start).total_seconds()) + 's')
    print(results_proba[:100])
    if self.ids is not None:
      self.estimates[:, 0] = self.ids
      self.estimates[:, 1] = results_proba[:, 1]
    else:
      self.crosscheck_estimates = {
        'labels': classifier.predict(self.test_data),
        'proba': results_proba[:, 1]
      }

    # inverted_vocab = {_id:w for (w,_id) in self.vectorizer.vocabulary_.items() }
    # for _id in classifier.coef_.argsort()[0][-50:]:
    #   print(inverted_vocab[_id], classifier.coef_[0][_id])

  def run(self):
    self.estimate()
    print(self.estimates)
    return self.estimates

  def run_crosscheck(self):
    self.estimate()
    print(self.crosscheck_estimates)
    return self.crosscheck_estimates

def run_process(X, Y, test_data, ids, vectorizer):
  runner = StrangeClassifier(X = X, Y = Y, test_data = test_data, ids = ids, vectorizer = vectorizer)
  return runner.run()

def run_process_crosscheck(X, Y, test_data, vectorizer):
  runner = StrangeClassifier(X = X, Y = Y, test_data = test_data, vectorizer = vectorizer)
  return runner.run_crosscheck()

def run_normalizer(vec):
  return normalize_array(vec)

if __name__ == '__main__':
  total_data_pts = 13944
  crosscheck = True
  stop_words = [
    'а',
    #'атьс',
    'ах',
    'бы',
    'быть',
    'в',
    'вать',
    'во',
    'вот',
    'всего',
    'всё',
    'вы',
    'для',
    'да',
    'до',
    'если',
    'ещё',
    'ение',
    'же',
    'за',
    'и',
    #'иват',
    'из',
    'ие',
    'ия',
    'или',
    'к',
    'ки',
    'как',
    'ко',
    'который',
    'кто',
    'ку',
    'ли',
    'лишь',
    'между',
    'мы',
    'на',
    'над',
    'нибудь',
    'никак',
    'нный',
    'но',
    'ну'
    #'нять',
    'о',
    'об',
    'овать',
    'оват',
    'ой',
    'ок',
    'около',
    'он',
    'она',
    'они',
    #'ость',
    'от',
    'по',
    'под',
    'практически',
    'при',
    'про',
    'просто',
    'с',
    #'сить',
    'совсем',
    'среди',
    #'ство',
    'так',
    'таки',
    'тать',
    'тем',
    'то',
    'ты',
    'ть',
    'тьс',
    'ться',
    'тот',
    'у',
    'уже',
    'чем',
    'что',
    'чтобы',
    'ься',
    #'ывать',
    #'ыват',
    'это',
    'этот',
    'src',
    'https',
    'figure'
  ]

  neg_stop_words = []
  for w in stop_words:
    neg_stop_words.append('не_' + w)

  stop_words = stop_words + neg_stop_words

  calc_start = dt.datetime.now()

  td = load_text_data(total_data_pts, True, filename='train_normalized_withspaces.csv')
  testd = load_text_data(total_data_pts, False, filename='test_normalized_withspaces.csv')
  (tv_norm, testv_norm, vectorizer_norm) = vectorize_data(
    td = td[0],
    testd = testd,
    stop_words = stop_words,
    vectorizer = TfidfVectorizer(
      stop_words = stop_words,
      ngram_range = (1, 3),
      max_df = .7,
      min_df = 3,
      #max_features = 96000
    ),
  )

  td = load_text_data(total_data_pts, True, filename='train_normalized_just4grams.csv')
  testd = load_text_data(total_data_pts, False, filename='test_normalized_just4grams.csv')
  (tv_grams, testv_grams, vectorizer_grams) = vectorize_data(
    td = td[0],
    testd = testd,
    stop_words = stop_words,
    vectorizer = TfidfVectorizer(
      stop_words = stop_words,
      ngram_range = (1, 2),
      max_df = .9,
      min_df = 3,
      # max_features = 16000
    ),
  )

  td = load_text_data(total_data_pts, True, filename='train_normalized_justpos.csv')
  testd = load_text_data(total_data_pts, False, filename='test_normalized_justpos.csv')
  (tv_pos, testv_pos, vectorizer_pos) = vectorize_data(
    td = td[0],
    testd = testd,
    stop_words = stop_words,
    vectorizer = TfidfVectorizer(
      stop_words = stop_words,
      ngram_range = (2, 4),
      max_df = .9,
      min_df = 3,
      #max_features = 16000
    ),
  )

  tv = hstack([tv_norm, tv_grams, tv_pos])
  testv = lil_matrix(hstack([testv_norm, testv_grams, testv_pos]))
  # tv = tv_norm
  # testv = testv_norm

  print('Vectorization time: ' + str((dt.datetime.now() - calc_start).total_seconds()) + 's')

  X_train, X_test, Y_train, Y_test = train_test_split(tv, td[1], test_size = .3)
  # X_test = X_train
  # Y_test = Y_train
  test_data_pts = Y_test.shape[0]
  # test_data_pts = Y_train.shape[0]
  current_vectorizer = None

  if crosscheck:
    print('WTF???', X_train.shape, Y_train.shape)
    # minimizer = TruncatedSVD(n_components=250, n_iter=100)
    # minimizer.fit(vstack([X_train, X_test]))
    # X_train_whooshed = minimizer.transform(X_train)
    # X_test_whooshed = minimizer.transform(X_test)
    #tmpClassifier = DecisionTreeClassifier(max_depth = 32, class_weight = { 0: 1, 1: 9 })
    #tmpClassifier = RandomForestClassifier(
    #  max_depth = 32,
    #  n_estimators = 64,
    #  max_features = 0.25,
    #  class_weight = { 0: 1, 1: 9 },
    #  n_jobs = 3
    #)
    #tmpClassifier = MultinomialNB(alpha = 0.02)
    # tmpClassifier = BernoulliNB()
    # tmpClassifier.fit(X_train, Y_train)
    feature_selector = SelectPercentile(chi2, percentile = 10)
    #feature_selector = SelectFromModel(tmpClassifier, prefit=True, threshold = 'median')
    feature_selector.fit(X_train, Y_train)
    X_train_whooshed = feature_selector.transform(X_train)
    X_test_whooshed = feature_selector.transform(X_test)
    print('Minification time: ' + str((dt.datetime.now() - calc_start).total_seconds()) + 's')
    (new_X, new_Y) = geld_data(X_train_whooshed, Y_train, current_vectorizer, StrangeClassifier)
  else:
    # minimizer = TruncatedSVD(n_components=250, n_iter=100)
    # minimizer.fit(vstack([tv, testv]))
    # X_train_whooshed = minimizer.transform(tv)
    # X_test_whooshed = minimizer.transform(testv)

    tmpClassifier = BernoulliNB()
    tmpClassifier.fit(tv, td[1])
    #feature_selector = SelectPercentile(mutual_info_classif, percentile = 25)
    feature_selector = SelectFromModel(tmpClassifier, prefit=True, threshold = 'median')
    #feature_selector.fit(X_train, Y_train)
    X_train_whooshed = feature_selector.transform(tv)
    X_test_whooshed = feature_selector.transform(testv)

    #X_train_whooshed = tv
    #X_test_whooshed = testv
    print('Minification time: ' + str((dt.datetime.now() - calc_start).total_seconds()) + 's')
    (new_X, new_Y) = geld_data(X_train_whooshed, td[1], current_vectorizer, StrangeClassifier)
  print('Culling time: ' + str((dt.datetime.now() - calc_start).total_seconds()) + 's')

  process_arrays = []
  data_arrays = []
  thread_cnt = 3 #1
  pool = multiprocessing.Pool(processes = thread_cnt)
  for cpu in range(thread_cnt):
    if crosscheck:
      start_index = cpu * test_data_pts // thread_cnt
      end_index = (cpu + 1) * test_data_pts // thread_cnt
      # start_index = cpu * total_data_pts // thread_cnt
      # end_index = (cpu + 1) * total_data_pts // thread_cnt

      process_arrays.append(pool.apply_async(run_process_crosscheck, [
        new_X,
        new_Y,
        # X_train[start_index:end_index],
        X_test_whooshed[start_index:end_index],
        # tv[start_index:end_index],
        current_vectorizer
      ]))
    else:
      start_index = cpu * total_data_pts // thread_cnt
      end_index = (cpu + 1) * total_data_pts // thread_cnt
      process_arrays.append(pool.apply_async(run_process, [
        new_X,
        new_Y,
        # tv[start_index:end_index],
        # td[0]['id'][start_index:end_index],
        X_test_whooshed[start_index:end_index],
        testd['id'][start_index:end_index],
        current_vectorizer
      ]))

  for p in process_arrays:
    data_arrays.append(p.get())

  print('Exec time: ' + str((dt.datetime.now() - calc_start).total_seconds()) + 's')

  if crosscheck:
    Y_pred = np.zeros(Y_test.shape)
    Y_proba = np.zeros(Y_test.shape)

    for idx, arr in enumerate(data_arrays):
      Y_pred[idx * test_data_pts // thread_cnt:(idx + 1) * test_data_pts // thread_cnt] = arr['labels']
      Y_proba[idx * test_data_pts // thread_cnt:(idx + 1) * test_data_pts // thread_cnt] = arr['proba']

    #print(Y_pred[:100], Y_test[:100])
    print(classification_report(y_true = Y_test, y_pred = Y_pred))
    print('roc_auc', roc_auc_score(y_true = Y_test, y_score = Y_proba))
    #print(classification_report(y_true = td[1], y_pred = Y_pred))
    #print('roc_auc', roc_auc_score(y_true = td[1], y_score = Y_proba))
  else:
    final_results = np.zeros((total_data_pts, 2))
    for idx, arr in enumerate(data_arrays):
      final_results[idx * total_data_pts // thread_cnt:(idx + 1) * total_data_pts // thread_cnt] = arr

    write_data(final_results)

