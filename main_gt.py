import json
from funcs_gt import *
import os

channel_list = ['express', 'hum', 'ptv']
channel_list_2 = ['express']

#*********************************************************#
#                       Main
#*********************************************************#
if __name__ == '__main__':
    """
    print(os.getcwd())
    lol = parse_anno_file("./Manual_Anno/Hum/hum_tv/annotations.xml", "2499.jpg")
    xtl = float(lol[0]['shapes'][0]['xtl'])
    print("xtl - 100: ", xtl - 100)
    print("XTL: ", lol[0]['shapes'][0]['xtl'])
    """

    for channel_name in channel_list_2:
        ground_truth = os.path.join(channel_name, "infered_results")
        xml_file = os.path.join(channel_name, "annotations.xml")
        #input_manual = os.path.join(channel_name, "images")
        print("**********************")
        print("Channel: ", channel_name)
        print("Ground Truth: ", ground_truth)
        print("XML File: ", xml_file)
        #print("Input Manual: ", input_manual, "\n")
        genrate_final_annotation(ground_truth, xml_file)
        #list_anno_file(xml_file)


    
    """
    #input_file = "./Manual_Anno/Hum/hum_tv_2/frames"
    ground_truth = "./Manual_Anno/Hum/hum_tv_2/infered_results"
    xml_file="./Manual_Anno/Hum/hum_tv_2/annotations.xml"
    input_manual = "./Manual_Anno/Hum/hum_tv_2/images"
    #match_annotations(ground_truth , xml_file , input_manual)
    genrate_final_annotation(ground_truth, xml_file, input_manual)
    #test_images(input_file , ground_truth , xml_file , input_manual)
    """