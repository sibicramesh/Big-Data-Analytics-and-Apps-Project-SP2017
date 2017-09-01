import tensorflow as tf, sys
import pymongo

mylist=[0,0,0,0]
mylabel=['deforestation','glaciermelting','sealevelrise','soilacidification']
#image_path = sys.argv[1]
for i in range(1,5):
 #print(i)
 image_path = 'data/climate/test/'+str(i)+'.jpg'
 print(image_path)
#image_path='data/climate/GlacierMelting/n09289331_20.JPEG'

# Read in the image_data
 image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
 label_lines = [line.rstrip() for line
               in tf.gfile.GFile("data/output_labels.txt")]

# Unpersists graph from file
 with tf.gfile.FastGFile("data/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

 with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))



    for node_id in top_k:

        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if human_string == 'deforestation':
            mylist[0] = mylist[0] + 1

        if human_string == 'glaciermelting':
            mylist[1] = mylist[1] + 1

        if human_string == 'sealevelrise':
            mylist[2] = mylist[2] + 1

        if human_string == 'soilacidification':
            mylist[3] = mylist[3] + 1

        #print(mylist[1])
        break
        #print(human_string)
    print("\n\n")


count = max(range(len(mylist)), key = lambda i: mylist[i])
client = pymongo.MongoClient('mongodb://sample:1234@ds059804.mlab.com:59804/api_database')
db = client['api_database']
test = db['test']
post_data = {
            'name': ''+mylabel[count]+''
        }
test.insert_one(post_data).inserted_id

