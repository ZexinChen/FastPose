import copy
import sys
import numpy as np

#point format: [x,y,confidence,point_index,class,pair_distance,min_distance]

def matchJudge(point_1,point_2,middlepoint_list,distance_tolerance=0.1):
    #point_1,point_2: (x,y,score,id)
    #middlepoint_list: [(x,y,score,id),(x,y,score,id)....]
    min_distance = sys.float_info.max
    match_count = 0
    match_middlepoints = [[]]
    for middlepoint in middlepoint_list:
        vector_point1_point2_x = point_2[0] - point_1[0]
        vector_point1_point2_y = point_2[1] - point_1[1]
        vector_point1_middlepoint_x = middlepoint[0] - point_1[0]
        vector_point1_middlepoint_y = middlepoint[1] - point_1[1]

        vector_point1_point2_square = vector_point1_point2_x*vector_point1_point2_x + \
                                      vector_point1_point2_y*vector_point1_point2_y
        vectror_dotProduct = vector_point1_point2_x*vector_point1_middlepoint_x+\
                             vector_point1_point2_y*vector_point1_middlepoint_y
        
        #calculate the distance between middlepoint and the line segment:point_1--point_2
        if vector_point1_point2_square>0:
            t = float(vectror_dotProduct)/vector_point1_point2_square
        else:
            return 0,[],min_distance
        if (t < 0):
            t = 0
        elif (t > 1):
            t = 1
        distance_x = point_1[0] + t*vector_point1_point2_x - middlepoint[0]
        distance_y = point_1[1] + t*vector_point1_point2_y - middlepoint[1]
        distance_point_vector = (distance_x*distance_x + distance_y*distance_y)**0.5

        # # #calculate the distance between middlepoint and the line middle point:point_1--point_2
        mp_x = (point_2[0] + point_1[0])/2
        mp_y = (point_2[1] + point_1[1])/2
        distance_point_middlepoint = ((middlepoint[0]-mp_x)*(middlepoint[0]-mp_x) + \
                                             (middlepoint[1]-mp_y)*(middlepoint[1]-mp_y))**0.5

        d_point1_point2 = vector_point1_point2_square**0.5
        if d_point1_point2<=10:
            if distance_point_middlepoint/ d_point1_point2 < distance_tolerance*3 and \
                distance_point_vector / d_point1_point2 < distance_tolerance*3:
                match_bool = True
            else:
                match_bool = False
        elif d_point1_point2<=55:#45:
            if distance_point_middlepoint / d_point1_point2 < distance_tolerance and \
                distance_point_vector / d_point1_point2 < distance_tolerance:
                match_bool = True
            else:
                match_bool = False
        else:
            if distance_point_middlepoint / d_point1_point2 < distance_tolerance and \
                distance_point_vector / d_point1_point2 < distance_tolerance*0.5:
                # ((distance_point_vector)) / (vector_point1_point2_square**0.5) < distance_tolerance*0.35:
                match_bool = True
            else:
                match_bool = False

        if match_bool:
            if  ((distance_point_middlepoint)) / (d_point1_point2) < min_distance:
                min_distance =  ((distance_point_middlepoint)) / (d_point1_point2)
                match_middlepoints[0] = middlepoint
            match_count+=1 
            # match_middlepoints.append(middlepoint)
           

    return match_count,match_middlepoints,min_distance

hip_dict = {}
def judgeHip(point_hip,person,jointLength_i,p_idx):
    global hip_dict
    right_shoulder = None
    left_shoulder = None
    left_hip = None
    right_hip = None
    for p_i in range(len(person)):
        if int(person[p_i][4])==2:
            right_shoulder = person[p_i]
        if int(person[p_i][4])==5:
            left_shoulder = person[p_i]
        if int(person[p_i][4])==8:
            right_hip = person[p_i]
        if int(person[p_i][4])==11:
            left_hip = person[p_i]
        if isClose(point_hip,person[p_i],hard=True):
            person[p_i].append(1)
            return False
        if len(person[p_i])>5 and int(person[p_i][4]) == int(point_hip[4]):
            return False

    if right_shoulder==None or left_shoulder==None:
        return True

    if point_hip[4]==8:#right hip
        x=np.array([left_shoulder[0]-right_shoulder[0],left_shoulder[1]-right_shoulder[1]])
        y=np.array([point_hip[0]-right_shoulder[0],point_hip[1]-right_shoulder[1]])
        hip_dis = pointDist(point_hip,right_shoulder)**0.5
        cos_angle = x.dot(y)/ ( np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))  )
        if right_hip != None:
            y_in=np.array([right_hip[0]-right_shoulder[0],right_hip[1]-right_shoulder[1]])
            cos_angle_in = x.dot(y_in)/ ( np.sqrt(x.dot(x)) * np.sqrt(y_in.dot(y_in))  )
            if abs(cos_angle_in)<abs(cos_angle):#the hip in persons is better
                return False
    else:#left hip
        x=np.array([right_shoulder[0]-left_shoulder[0],right_shoulder[1]-left_shoulder[1]])
        y=np.array([point_hip[0]-left_shoulder[0],point_hip[1]-left_shoulder[1]])
        hip_dis = pointDist(point_hip,left_shoulder)**0.5
        cos_angle = x.dot(y)/ ( np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))  )
        if left_hip != None:
            y_in=np.array([left_hip[0]-right_shoulder[0],left_hip[1]-right_shoulder[1]])
            cos_angle_in = x.dot(y_in)/ ( np.sqrt(x.dot(x)) * np.sqrt(y_in.dot(y_in))  )
            if abs(cos_angle_in)<abs(cos_angle):#the hip in persons is better
                return False
    #the following is the case where not better hip in person
    shoulder_dis = pointDist(right_shoulder,left_shoulder)**0.5
    jointLength_i_max = max(jointLength_i)**0.5

    if point_hip==[123.0, 165.0, 0.4263727702200413, 36.0, 8.0] or point_hip==[548.0, 178.0, 0.17753265870356927, 37.0, 8.0]:
        print('judging hip',point_hip,\
            hip_dis,\
            shoulder_dis,\
            jointLength_i_max,\
            cos_angle,\
            hip_dis>shoulder_dis*2,hip_dis>jointLength_i_max*2,(cos_angle<-0.5 or cos_angle>1/2**0.5),\
            '1:',(shoulder_dis>10 and (cos_angle<-0.5 or cos_angle>1/2**0.5) and hip_dis>jointLength_i_max*2),\
            '2:',(shoulder_dis>5 and (cos_angle<-0.8 or cos_angle>0.8) and hip_dis>jointLength_i_max*10),\
            '3:',(len(jointLength_i)>3 and shoulder_dis>5 and hip_dis>jointLength_i_max*20),\
            '4:',(hip_dis>shoulder_dis*20 and (cos_angle<-0.8 or cos_angle>0.8)),\
            '5:',(('%d-%d'%(p_idx,int(point_hip[4]))) in hip_dict and abs(cos_angle-0.01)>hip_dict['%d-%d'%(p_idx,int(point_hip[4]))])
            )
    if shoulder_dis>10 and (cos_angle<-0.5 or cos_angle>1/2**0.5) and hip_dis>jointLength_i_max*2:
        return False
    elif shoulder_dis>5 and (cos_angle<-0.8 or cos_angle>0.8) and hip_dis>jointLength_i_max*10:
        return False
    elif len(jointLength_i)>3 and shoulder_dis>5 and hip_dis>jointLength_i_max*20:
        return False
    elif shoulder_dis>5 and hip_dis>shoulder_dis*20 and (cos_angle<-0.8 or cos_angle>0.8):
        return False
    elif ('%d-%d'%(p_idx,int(point_hip[4]))) in hip_dict and abs(cos_angle-0.01)>abs((hip_dict['%d-%d'%(p_idx,int(point_hip[4]))])-0.01):
        return False
    else:
        hip_dict['%d-%d'%(p_idx,int(point_hip[4]))] = cos_angle
        return True

def addMatchPoint(point_1,point_2,match_middlepoints_max,persons,persons_withMiddlePoint,jointLength):
    added = False
    i=-1
    for i in range(len(persons_withMiddlePoint)):
        if not (int(point_1[4])==11 and int(point_2[4])==8)\
            and inlist(point_1,persons[i]): # not left right hip
            persons_withMiddlePoint[i].append(point_2)
            persons_withMiddlePoint[i].append(match_middlepoints_max[0])
            persons[i].append(point_2)
            jointLength[i].append(pointDist(point_1,point_2))
            added = True
            add_type = 'add point_2'
            break
        if int(point_1[4])==11 and int(point_2[4])==8 and (inlist(point_2,persons[i]) or inlist(point_1,persons[i])):

            if inlist(point_2,persons[i]) and judgeHip(point_1,persons[i],jointLength[i],i):
                persons_withMiddlePoint[i].append(point_1)
                persons_withMiddlePoint[i].append(match_middlepoints_max[0])
                persons[i].append(point_1)
                jointLength[i].append(pointDist(point_1,point_2))
                added = True
                add_type = 'add point_1'
            if inlist(point_1,persons[i]) and judgeHip(point_2,persons[i],jointLength[i],i):
                persons_withMiddlePoint[i].append(point_2)
                persons_withMiddlePoint[i].append(match_middlepoints_max[0])
                persons[i].append(point_2)
                jointLength[i].append(pointDist(point_1,point_2))
                added = True
                add_type = 'add point_2'

            break
        
    if not added:
        persons_withMiddlePoint.append([point_1,point_2,match_middlepoints_max[0]])
        persons.append([point_1,point_2])
        jointLength.append([0,pointDist(point_1,point_2)])
        add_type = 'add point_1 and point_2(new list)'

    return persons,persons_withMiddlePoint

def addPair(point_1,point_2,match_middlepoints,pair):
    pair.append([point_1,point_2,match_middlepoints[0]])
    return pair

def justifyPoint(point_1,point_2,match_middlepoints,jointLength,persons,min_distance_in,min_distance,match_lenDistance_min,distance_tolerance=1.5):
    person_id2 = -1
    if ((int(point_2[4]) in [2]) and (int(point_1[4]) in [5])):# right shoulder and left shoulder
        person_id = findPersonIdx(match_middlepoints[0],persons)
    elif ((int(point_2[4]) in [8]) and (int(point_1[4]) in [11])): # right hip and left hip
        person_id = findPersonIdx(point_1,persons)
        person_id2 = findPersonIdx(point_2,persons)
    else:
        person_id = findPersonIdx(point_1,persons)

    place = ''
    if (person_id==len(persons) and (person_id2 == -1 or person_id2==len(persons))):# new person
        if point_1==[251.0, 99.0, 0.30553005053661764, 49.0, 5.0] and point_2== [199.0, 86.0, 0.21249479273683392, 19.0, 2.0]:
            print('hear')
        if int(point_2[4]) in [0,1,14,15,16,17] and pointDist(point_1,point_2)**0.5>150:#head
            return_value = False
        elif int(point_2[4]) in [2] and int(point_1[4]) in [5] \
            and pointDist(point_1,point_2)**0.5>200 and min_distance_in>0.01:# right shoulder and left shoulder
            place = '1'
            return_value = False
        else:
            return_value = True
    elif int(point_2[4]) in [1]:#neck
        if pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*10:
            return_value = False
        else:
            return_value = True
    elif ((int(point_2[4]) in [2]) or (int(point_2[4]) in [5])):# right shoulder and left shoulder
        if max(jointLength[person_id])**0.5<10 and pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*5:
            place = '21'
            return_value = False
        elif max(jointLength[person_id])**0.5>10 and pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*2.5:
            place = '22:'+str(max(jointLength[person_id])**0.5)
            return_value = False
        else:
            return_value = True
    elif int(point_2[4]) in [0,14,15,16,17]:#head 
        if max(jointLength[person_id])**0.5>10 and pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*3:
            return_value = False
        elif pointDist(point_1,point_2)**0.5>50 and pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*3:
            return False
        else:
            return_value = True
    elif int(point_2[4]) in [8,11]:#hip 
        if ((int(point_2[4]) in [8]) and (int(point_1[4]) in [11])): # right hip and left hip
            # print(person_id,person_id2,len(persons))
            return_value = (person_id!=len(persons) and judgeHip(point_2,persons[person_id],jointLength[person_id],person_id)) \
                            or (person_id2!=len(persons) and judgeHip(point_1,persons[person_id2],jointLength[person_id2],person_id2))
        else:
            return_value = judgeHip(point_2,persons[person_id],jointLength[person_id],person_id)
        if point_1==[196.0, 157.0, 0.5946665434166789, 18.0, 2.0] and point_2==[211.0, 279.0, 0.4176066023974272, 85.0, 8.0]:
            print('hip judged:',point_1,point_2,return_value)
    elif int(point_2[4]) in [3,4,6,7]:#elbow and hand
        if pointDist(point_1,point_2)**0.5> 10 \
           and pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*3:
            return_value = False
        elif pointDist(point_1,point_2)**0.5<= 10\
           and pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*5:
            return_value = False
        else:
            return_value = True
    elif int(point_2[4]) in [9,12]:#knee
        if abs(pointDist(point_1,point_2))**0.5>match_lenDistance_min**0.5*1.2:
            return_value = False 
        elif max(jointLength[person_id])**0.5>10 and pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*3:
            return_value = False
        else:
            return_value = True
    else:
        if pointDist(point_1,point_2)**0.5 > max(jointLength[person_id])**0.5*1.5:
            return_value = False
        else:
            return_value = True
        #neck

    if return_value == True:
        if int(point_2[4]) in [0,1,14,15,16,17]:#head 
            if abs(pointDist(point_1,point_2))>match_lenDistance_min:
                return_value = False
        else:
            if abs(pointDist(point_1,point_2))**0.5<50 and min_distance_in>min_distance+0.03:
                return_value = False
            elif abs(pointDist(point_1,point_2))**0.5>=50 and min_distance_in>min_distance+0.015:
                place = '2'
                return_value = False
            else:
                if abs(pointDist(point_1,point_2))>match_lenDistance_min*4:
                    return_value = False

    if point_1==[348.0, 374.0, 0.6267240168526769, 96.0, 9.0]\
        and (point_2== [363.0, 430.0, 0.5354817640036345, 107.0, 10.0]\
        or  point_2 ==[363.0, 430.0, 0.5354817640036345, 107.0, 10.0]):
        if person_id==len(persons):# new person
            print('justify(person_id==len(jointLength)):',point_1,point_2,match_middlepoints[0],\
                pointDist(point_1,point_2)**0.5,match_lenDistance_min**0.5,\
                min_distance_in,min_distance,\
                return_value,'!!!!!!',(person_id==len(persons) and (person_id2 == -1 or person_id2==len(persons))))
        else:
            print('justify:',point_1,point_2,match_middlepoints[0])
            print(pointDist(point_1,point_2)**0.5,match_lenDistance_min**0.5,\
                min_distance_in,min_distance,\
                max(jointLength[person_id])**0.5,\
                (person_id!=len(jointLength) \
            and abs(match_lenDistance_min**0.5-max(jointLength[person_id])**0.5)\
             >2*abs(abs(pointDist(point_1,point_2))**0.5-max(jointLength[person_id])**0.5)\
            and min_distance_in<min_distance+0.1),\
                abs(match_lenDistance_min**0.5-max(jointLength[person_id])**0.5),\
                abs(abs(pointDist(point_1,point_2))**0.5-max(jointLength[person_id])**0.5),\
                return_value, 
                )


    return return_value

def findBetter(match_point,point_1,peaks_1,peaks_middlePoint,min_distance,distance_tolerance,jointLength,persons):

    for p1 in peaks_1:
        match_count,match_middlepoints,min_distance_in = matchJudge(match_point,p1,
                                                    peaks_middlePoint,
                                                    distance_tolerance=distance_tolerance)
        if match_count>0:
            if abs(pointDist(point_1,match_point)) > abs(pointDist(p1,match_point))*1.1\
                and min_distance_in < min_distance+0.015:# or min_distance_in*1.01<min_distance:
                if match_point==[254.0, 366.0, 0.13061288077187783, 22.0, 9.0]:
                    print('findbetter',\
                        point_1,p1,match_point,match_count,match_middlepoints,\
                        min_distance_in,min_distance,\
                        abs(pointDist(point_1,match_point)) , abs(pointDist(p1,match_point))*1.1)
                return True

    return False

def person_group_test(all_peaks,all_peaks_vec,distance_tolerance=0.2):
    coco_ann = { 'nose':0, 'neck':1, 'right_shoulder' :2, 'right_elbow' :3, 'right_wrist':4, 
             'left_shoulder':5, 'left_elbow':6, 'left_wrist':7, 'right_hip':8, 
             'right_knee':9, 'right_ankle':10, 'left_hip':11, 'left_knee':12, 
             'left_ankle':13, 'right_eye':14, 'left_eye':15, 'right_ear':16, 
             'left_ear':17 }
    coco_middle = {'nose--left_eye':0,'nose--right_eye':1,'left_eye--left_ear':2,'right_eye--right_ear':3,
               # 'left_shoulder--right_shoulder':4,
               'nose--neck':5,'left_shoulder--left_elbow':6,
               'right_shoulder--right_elbow':7,'left_elbow--left_wrist':8,'right_elbow--right_wrist':9,
               'left_hip--right_hip':10,'left_shoulder--left_hip':11,'right_shoulder--right_hip':12,
               'left_hip--left_knee':13,'right_hip--right_knee':14,'left_knee--left_ankle':15,'right_knee--right_ankle':16,
               'neck--left_shoulder':4,'neck--right_shoulder':17}

    checklist = [
                 {'nose':['left_eye','right_eye','neck']},
                 {'left_eye':['left_ear']},
                 {'right_eye':['right_ear']},
                 {'neck':['left_shoulder','right_shoulder']},
                 {'left_shoulder':['left_elbow']},
                 {'right_shoulder':['right_elbow']},
                 {'left_elbow':['left_wrist']},
                 {'right_elbow':['right_wrist']},
                 {'left_shoulder':['left_hip']},
                 {'right_shoulder':['right_hip']},
                 {'left_hip':['right_hip','left_knee']},
                 {'right_hip':['right_knee']},
                 {'left_knee':['left_ankle']},
                 {'right_knee':['right_ankle']},
        ]
    person_last_len = [0 for ii in range(1000)]

    global hip_dict
    hip_dict = {}
    persons = []
    persons_withMiddlePoint = []
    pair = []
    jointLength = []


    for i in range(len(checklist)):
        peaks_1_key = list(checklist[i].keys())[0]
        peaks_1 = all_peaks[ coco_ann[peaks_1_key] ]
        for peaks_2_key in checklist[i][peaks_1_key]:
            peaks_2 = all_peaks[ coco_ann[peaks_2_key] ]
            for point_1 in peaks_1:
                last_len = person_last_len[findPersonIdx(point_1,persons)]
                match_lenDistance_min = sys.maxsize
                match_middlepoints_max = None
                match_point = []
                min_distance = sys.maxsize
                for point_2 in peaks_2:
                    match_count,match_middlepoints,min_distance_in = matchJudge(point_1,point_2,
                                                                all_peaks_vec[coco_middle[peaks_1_key+'--'+peaks_2_key]],
                                                                distance_tolerance=distance_tolerance)
                    
                
                    if match_count>0 and justifyPoint(point_1,point_2,match_middlepoints,jointLength,persons,min_distance_in,min_distance,match_lenDistance_min):
                        
                        min_distance = min_distance_in
                        match_lenDistance_min = abs(pointDist(point_1,point_2))
                        match_middlepoints_max = match_middlepoints
                        match_point = point_2

                if match_point==[323.0, 383.0, 0.29623862297739834, 56.0, 12.0]:
                    print('found better?',findBetter(match_point,point_1,peaks_1,all_peaks_vec[coco_middle[peaks_1_key+'--'+peaks_2_key]],min_distance,distance_tolerance,jointLength,persons))
                if len(match_point)!=0 and not findBetter(match_point,point_1,peaks_1,all_peaks_vec[coco_middle[peaks_1_key+'--'+peaks_2_key]],min_distance,distance_tolerance,jointLength,persons): 
                    
                    person_last_len[findPersonIdx(point_1,persons)] = match_lenDistance_min
                    persons,persons_withMiddlePoint = addMatchPoint(point_1,match_point,match_middlepoints_max,persons,persons_withMiddlePoint,jointLength)
                    

    return persons,pair,persons_withMiddlePoint,jointLength,peaks_1,peaks_2,all_peaks_vec[coco_middle[peaks_1_key+'--'+peaks_2_key]]
 
def removeDouble(persons,jointLength):
    new_person = copy.deepcopy(persons)
    set_remove = []
    for i in range(len(persons)):
        for pt_i in range(len(persons[i])):
            for j in range(i+1,len(persons)):
                for pt_j in range(len(persons[j])):
                    if isClose(persons[i][pt_i],persons[j][pt_j]):
                        if jointLength[i][pt_i]>jointLength[j][pt_j]:
                            if not [i,pt_i] in set_remove:
                                set_remove.append([i,pt_i])
                        else:
                            if not [j,pt_j] in set_remove:
                                set_remove.append([j,pt_j])

    for sr in set_remove:
        while myIn(persons[sr[0]][sr[1]] , new_person[sr[0]])!=sys.maxsize:
            del new_person[sr[0]][myIn(persons[sr[0]][sr[1]] , new_person[sr[0]])]


    return new_person

def pointDist(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2

def myIn(ele,li):
    for i in range(len(li)):
        if np.all(ele == li[i]):
            return i
    return sys.maxsize

def inlist(point,point_list):

    for p in point_list:

        if isClose(point,p):
            if point[4]==p[4]:
                return True
            if int(point[4]) in [1] and int(p[4]) in [4,4+18]:
                return True
            if int(p[4]) in [1] and int(point[4]) in [4,4+18]:
                return True

    return False

def findPersonIdx(point,persons):
    for i in range(len(persons)):
        if inlist(point, persons[i]):
            return i

    return len(persons)

def isClose(p1,p2,hard=False):

    if hard:
        return (p1[0]-p2[0])**2<9 and (p1[1]-p2[1])**2<9 and int(p1[4])==int(p2[4])
    else:
        return (p1[0]-p2[0])**2<9 and (p1[1]-p2[1])**2<9 
