import json

from numpy import size
from funcs_stat import *
import os

channel_list = ['express', 'hum', 'ptv', 'samaa']
channel_list_2 = ['express']

#*********************************************************#
#                       Main
#*********************************************************#

size_arr = np.arange(20, 85, 5)
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
"""
#size_arr = [30]
#pose_arr = [[70, 70, 50]]

express_scores = []

small_count = np.zeros(len(size_arr))
size_acc = np.zeros(len(size_arr))
total_size = np.zeros(len(size_arr))

invalid_count = np.zeros(len(pose_arr))
pose_acc = np.zeros(len(pose_arr))
total_pose = np.zeros(len(pose_arr))

#print("Length of Size Array: ", len(size_arr))

if __name__ == '__main__':

    hp_iter_counter = 0
    total_considered = 0

    legend_str = ""
    for base_size in size_arr:
        for base_pose in pose_arr:
            legend_str = legend_str + str(hp_iter_counter) + ": base_size = [" + str(base_size) + ", " + str(base_size) + "]" + ",   base_pose = " + str(base_pose) + "\n\n"
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

                small_faces = 0
                invalid_poses = 0
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

                    lmao = 0
                    
                    for i in range(len(pred_data['Bbox'])):
                        validated = 0
                        pred_box = pred_data['Bbox'][i]
                        anot_ind = 0
                        grt_IOU = 0
                        for j in range(len(anot_data['Bbox'])):
                            anot_box = anot_data['Bbox'][j]
                            #print("Pred Box: ", pred_box)
                            #print("Anot Box: ", anot_box)
                            IOU = bb_intersection_over_union(pred_box, anot_box)
                            #print("Iou for ", pred_data['Label'][i], " and ", anot_data['Label'][j], " :", IOU)
                            #print("IOU: ", IOU)
                            if IOU > 0.35:
                                if IOU > grt_IOU:
                                    validated = 1
                                    anot_ind = j
                                    grt_IOU = IOU
                                #print("Truly Positive")
                        if validated == 1:
                            det_TP += 1
                            det_local_tp += 1
                            
                            pred_label = pred_data['Label'][i]
                            anot_comparator = anot_data['Label'][anot_ind]

                            #print("Matched Pred: ", pred_label)
                            #print("Matched Anot: ", anot_comparator)
                            
                            #print("Recognition Validated for Label: ", pred_label)
                            rec_val = 0

                            given_size = pred_data['size'][i]
                            given_pose = pred_data['Pose'][i]

                            size_val = calcSize(base_size, given_size)
                            pose_val = calcPose(base_pose, given_pose)

                            if size_val == 0:
                                small_faces += 1
                            if pose_val == 0:
                                invalid_poses += 1
                            
                            if pred_label != 'small face' and pred_label != 'Invalid Pose' and size_val == 1 and pose_val == 1:
                                if pred_label == anot_comparator:
                                    rec_local_tp += 1
                                    rec_TP += 1
                                elif pred_label == 'Unknown':
                                    local_rec_FN += 1
                                else:
                                    local_rec_FP += 1

                    rec_FN = rec_FN + local_rec_FN
                    rec_FP = rec_FP + local_rec_FP
                    
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
                
                rec_acc = rec_TP / (rec_TP + rec_FP + rec_FN)

                total_considered = total_considered + (rec_TP + rec_FP + rec_FN)

                small_face_perc = small_faces / (rec_TP + rec_FP + rec_FN + small_faces)
                small_face_perc = small_face_perc * 100

                invalid_pose_perc = invalid_poses / (rec_TP + rec_FP + rec_FN + invalid_poses)
                invalid_pose_perc = invalid_pose_perc * 100

                pitch = base_pose[0]
                index = (pitch - 45) / 5
                index = len(pose_arr) - index - 1
                index = int(index)

                #print("Pose Index: ", index)

                invalid_count[index] = invalid_count[index] + invalid_poses
                pose_acc[index] = pose_acc[index] + rec_TP
                total_pose[index] = total_pose[index] + rec_TP + rec_FP + rec_FN + invalid_poses

                pitch = base_size
                index = (pitch - 20) / 5
                index = index 
                index = int(index)

                #print("Size Index: ", index)
                #print("Len Small Count: ", len(small_count))

                small_count[index] = small_count[index] + small_faces
                size_acc[index] = size_acc[index] + rec_TP
                total_size[index] = total_size[index] + rec_TP + rec_FP + rec_FN + small_faces



                append_wala = [rec_acc, base_size, base_pose, small_face_perc, invalid_pose_perc]

                if channel_name == 'express':
                    express_scores.append(append_wala)

                print("Small Faces: ", small_faces)
                print("Invalid Poses: ", invalid_poses)

                
                det_dict = {"True Positives" : det_TP, "False Positives" : det_FP, "False Negatives" : det_FN, "Precision" : det_precision, "Recall" : det_recall, "F1 Score" : det_f1s}
                rec_dict = {"True Positives" : rec_TP, "False Positives" : rec_FP, "False Negatives" : rec_FN, "Precision" : rec_precision, "Recall" : rec_recall, "F1 Score" : rec_f1s}
                
                dict = {"Channel" : channel_name, "Key" : hp_iter_counter, "Size" : int(base_size), "Pose" : base_pose, "Small Faces" : small_faces, "Invalid Poses" : invalid_poses, "Recognition Metrics" : rec_dict, "Detection Metrics" : det_dict}
                #print(dict)

                saver_string = channel_name + ".txt"
                if os.path.exists(saver_string):
                    #print("Updating File")
                    g = open(saver_string)
                    welp = g.read()
                    #print("Previous File: ", welp)
                    welp = str(welp) + "\n\n" + str(dict)
                    #print("Updated File: ", welp)
                    with open(saver_string, 'w') as f:
                        f.write(welp)
                else:
                    #print("Creating File")
                    with open(saver_string, 'w') as f:
                        f.write(str(dict))
            
            hp_iter_counter += 1

            #print("Defaulted Frames: ", defaulted_frames)
            #print("Number of Defaulted Frames: ", len(defaulted_frames))
            with open("problem_frames.txt", 'w') as f:
                f.write(str(defaulted_frames))

    #express_scores = [[12, 34], [45, 67], [98, 75]]
    #print("Express Scores: ", express_scores)
    with open("legend.txt", 'w') as f:
        f.write(legend_str)
    #print("Legend String: \n", legend_str)
    print("Total Considered: ", total_considered)
    print("Size Array: ", size_arr)
    print("Small Count: ", small_count)
    print("Size Accuracy: ", size_acc / total_size)#, "\n\n")
    print("Total Size: ", total_size, "\n\n")


    print("Pose Array: ", pose_arr)
    pose_arr_mod = [i[0] for i in pose_arr]
    print("Modified Pose Array: ", pose_arr_mod)
    print("Invalid Count: ", invalid_count / total_pose)
    print("Pose Accuracy: ", pose_acc / total_pose)
    #print("Total Pose: ", total_pose)

    plot_stat(size_arr, ((small_count / total_size) * 100), channel_list_2[0], size=1, acc=0)
    plot_stat(size_arr, ((size_acc / (total_size - small_count)) * 100), channel_list_2[0], size=1, acc=1)
    plot_stat(pose_arr_mod, ((invalid_count / total_pose) * 100), channel_list_2[0], size=0, acc=0)
    plot_stat(pose_arr_mod, ((pose_acc / (total_pose - invalid_count)) * 100), channel_list_2[0], size=0, acc=1)
   
