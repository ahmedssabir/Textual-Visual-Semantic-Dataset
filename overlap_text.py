
file1 = []

file2 = []

with open('train_visual.txt','rU') as f:
    for line in f:
       file1.append(line.rstrip())


with open('caption_anot.txt') as f1:
    for line1 in f1:
       file2.append(line1.rstrip())
       #break

f=open('intersection_caption_visual.txt', "w")
for i in range(len(file1)):
    temp =[]
    messages  = file1[i]
    messages1 = file2[i]

    words1 = messages.lower().split()
    words2 = messages1.lower().split()

    w = set(words1) & set(words2)


	#words1 = "This is a simple test of set intersection".lower().split()
	#words2 = "Intersection of sets is easy using Python".lower().split()

    
    temp.append(w)

    result= file1[i]+','+file2[i]+','+str(w)

    f.write(result)
    #f.write(result)
    f.write('\n')
    print(result)
    #del result
    #close.sess()
    
f.close()




