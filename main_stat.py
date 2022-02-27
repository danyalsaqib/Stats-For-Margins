import json
from funcs_stat import *
import os

channel_list = ['express', 'hum', 'ptv', 'samaa']
channel_list_2 = ['express']

#*********************************************************#
#                       Main
#*********************************************************#
"""
size_arr = np.arange(30, 65, 5)
#print(size_arr)
pose_arr = []
for i in range(6):
    lol = 70 - (i*5)
    lol2 = lol - 20
    lol3 = [lol, lol, lol2]
    pose_arr.append(lol3)
#print(pose_arr)
"""
size_arr = np.arange(30, 60, 10)
#print(size_arr)
pose_arr = []
for i in range(3):
    lol = 70 - (i*10)
    lol2 = lol - 20
    lol3 = [lol, lol, lol2]
    pose_arr.append(lol3)
#print(pose_arr)
#size_arr = [30]
#pose_arr = [[60, 60, 40]]
express_scores = []
hum_scores = []
ptv_scores = []
samaa_scores = []

if __name__ == '__main__':

    hp_iter_counter = 0
    for base_size in size_arr:
        for base_pose in pose_arr:            
            defaulted_frames = []
            for channel_name in channel_list_2:
                preds = os.path.join(channel_name, "infered_results")
                annotated = os.path.join(channel_name, "infered_results_final")
                print("\n**********************\n")
                print("Channel: ", channel_name)
                print("Size Limit: ", base_size)
                print("Pose Limit: ", base_pose)
                det_TP = 0
                det_FP = 0
                det_FN = 0
                rec_TP = 0
                rec_FP = 0
                rec_FN = 0
                for filename in os.listdir(preds):
                    with open(os.path.join(preds, filename)) as f:
                        pred_data = json.load(f)
                    with open(os.path.join(annotated, filename)) as g:
                        anot_data = json.load(g)
                    #print("File: ", filename)
                    #print("Prediction: ", len(pred_data['Label']))
                    #print("Annotation: ", len(anot_data['Label']))
                    missed = 0
                    det_local_tp = 0
                    rec_local_tp = 0
                    
                    unk_pred = 0
                    unk_anot = 0
                    for anot_counter_label in anot_data['Label']:
                        if anot_counter_label == 'Unknown':
                            unk_anot += 1

                    #print("Unk_Anot: ", unk_anot)

                    for pred_counter_label in pred_data['Label']:
                        if pred_counter_label == 'Unknown':
                            unk_pred += 1
                    
                    unk_counter = 0
                    local_rec_FP = 0
                    local_rec_FN = 0
                    rejecc = 0
                    
                    for i in range(len(pred_data['Bbox'])):
                        validated = 0
                        pred_box = pred_data['Bbox'][i]
                        for anot_box in anot_data['Bbox']:
                            IOU = bb_intersection_over_union(pred_box, anot_box)
                            #print("IOU: ", IOU)
                            if IOU > 0.35:
                                validated = 1
                                #print("Truly Positive")
                        if validated == 1:
                            det_TP += 1
                            det_local_tp += 1
                            
                            pred_label = pred_data['Label'][i]
                            #print("Recognition Validated for Label: ", pred_label)
                            rec_val = 0

                            given_size = pred_data['Size'][i]
                            given_pose = pred_data['Pose'][i]

                            size_val = calcSize(base_size, given_size)
                            pose_val = calcPose(base_pose, given_pose)


                            if pred_label != 'small face' and pred_label != 'Invalid Pose' and size_val == 1 and pose_val == 1:
                                #print("Successful File: ", filename)
                                #print("Successful Label: ", pred_label)
                                for anot_label in anot_data['Label']:
                                    if pred_label == anot_label:
                                        #print("Pred Label: ", pred_label)
                                        #print("Anot Label: ", anot_label)
                                        if pred_label == 'Unknown':
                                            if unk_counter < unk_anot:
                                                #print("Validated")
                                                unk_counter += 1
                                                rec_val = 1
                                                break
                                            else:
                                                #print("Unknowns exceeded")
                                                #print("FP - frame: ", pred_label)
                                                rec_FP += 1
                                                local_rec_FP += 1
                                                rec_val = 0
                                            #    rec_FP += 1
                                                break

                                        else:
                                            #print("Validated")
                                            rec_val = 1
                                            break
                            
                                if rec_val == 1:
                                    rec_local_tp += 1
                                    rec_TP += 1
                                #else:
                                #    rec_FP += 1
                                #    local_rec_FP += 1
                            else:
                                rejecc += 1
                    
                    #print("Unk_Count: ", unk_counter)
                    len_percieved_anot = 0
                    #print("Rejecc: ", rejecc)

                    for anot_counter_label in anot_data['Label']:
                        if anot_counter_label != 'small face' and anot_counter_label != 'Invalid Pose':
                            len_percieved_anot += 1
                    #print("Length Perceived: ", len_percieved_anot)
                    len_percieved_anot = len_percieved_anot - rejecc
                    rec_FN = rec_FN + len_percieved_anot - rec_local_tp
                    local_rec_FN = len_percieved_anot - rec_local_tp
                    if local_rec_FP > 0:
                        rec_FP = rec_FP - local_rec_FN

                                    
                    #print("Pred Boxes Length", len(pred_data['Bbox']))
                    #print("Pred Boxes: ", pred_data['Label'])
                    #print("Anot Boxes Length", len(anot_data['Bbox']))
                    #print("Anot Boxes: ", anot_data['Label'])
                    
                    local_det_FP = len(pred_data['Bbox']) - det_local_tp
                    local_det_FN = len(anot_data['Bbox']) - det_local_tp

                    if local_det_FP > 0 or local_det_FN > 0 or local_rec_FP > 0 or local_rec_FN > 0:
                        defaulter_dict = {channel_name, filename}
                        defaulted_frames.append(defaulter_dict)


                    det_FP = det_FP + len(pred_data['Bbox']) - det_local_tp
                    det_FN = det_FN + len(anot_data['Bbox']) - det_local_tp
                
                print("\nDetection")
                print("True Positives: ", det_TP)
                print("False Positives: ", det_FP)
                print("False Negatives: ", det_FN)

                det_precision = det_TP / (det_TP + det_FP)
                det_recall = det_TP / (det_TP + det_FN)
                det_f1s = 2 * ((det_precision * det_recall) / (det_precision + det_recall))

                print("Precision: ", det_precision)
                print("Recall: ", det_recall)
                print("F1 Score: ", det_f1s)

                print("\nRecognition")
                print("True Positives: ", rec_TP)
                print("False Positives: ", rec_FP)
                print("False Negatives: ", rec_FN)

                if (rec_TP + rec_FP) > 0 and (rec_TP + rec_FN) > 0:
                    rec_precision = rec_TP / (rec_TP + rec_FP)
                    rec_recall = rec_TP / (rec_TP + rec_FN)
                    rec_f1s = 2 * ((rec_precision * rec_recall) / (rec_precision + rec_recall))
                else:
                    rec_precision = 0
                    rec_recall = 0
                    rec_f1s = 0
                    

                print("Precision: ", rec_precision)
                print("Recall: ", rec_recall)
                print("F1 Score: ", rec_f1s)

                append_wala = [rec_precision, rec_recall]

                if channel_name == 'express':
                    express_scores.append(append_wala)
                elif channel_name == 'hum':
                    hum_scores.append(append_wala)
                elif channel_name == 'ptv':
                    ptv_scores.append(append_wala)
                elif channel_name == 'samaa':
                    samaa_scores.append(append_wala)

                
                det_dict = {"True Positives" : det_TP, "False Positives" : det_FP, "False Negatives" : det_FN, "Precision" : det_precision, "Recall" : det_recall, "F1 Score" : det_f1s}
                rec_dict = {"True Positives" : rec_TP, "False Positives" : rec_FP, "False Negatives" : rec_FN, "Precision" : rec_precision, "Recall" : rec_recall, "F1 Score" : rec_f1s}
                
                dict = {"Channel" : channel_name, "Size" : int(base_size), "Pose" : base_pose, "Detection Metrics" : det_dict, "Recognition Metrics" : rec_dict}
                #print(dict)

                saver_string = channel_name + ".txt"
                if os.path.exists(saver_string):
                    print("Updating File")
                    g = open(saver_string)
                    welp = g.read()
                    #print("Previous File: ", welp)
                    welp = str(welp) + "\n\n" + str(dict)
                    #print("Updated File: ", welp)
                    with open(saver_string, 'w') as f:
                        f.write(welp)
                else:
                    print("Creating File")
                    with open(saver_string, 'w') as f:
                        f.write(str(dict))

            #print("Defaulted Frames: ", defaulted_frames)
            #print("Number of Defaulted Frames: ", len(defaulted_frames))
            with open("problem_frames.txt", 'w') as f:
                f.write(str(defaulted_frames))

    print("Express Scores: ", express_scores)
    express_scores = [[12, 34], [45, 67], [98, 75]]
    plot_stat(express_scores, 'express')


