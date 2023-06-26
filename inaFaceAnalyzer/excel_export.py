#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:22:00 2023

@author: ddoukhan
"""

import xlsxwriter
import cv2
import os
from io import BytesIO
from .opencv_utils import imread_rgb
from .face_preprocessing import preprocess_face

def excel_export(df, dst, imcol):
    """
    df : pandas dataframe corresponding to an analysis
    dst : output excel filename
    imcol : name of the column containing the path to the image to display
    """
    cols = list(df.columns)
   
   
    # Create an new Excel file and add a worksheet.
    workbook = xlsxwriter.Workbook(dst)
    worksheet = workbook.add_worksheet()

    # Widen the first column to make the text clearer.
    worksheet.set_column('A:A',15)
    worksheet.set_default_row(80)

   
    worksheet.write(0, 0, 'face_image')
    for i, col in enumerate(cols):
        #print('col', col)
        worksheet.write(0, i+1, col)
   
    for ituple, t in enumerate(df.itertuples(index=False)):
        #print('line', ituple)
        for ielt, elt in enumerate(t):
            #print('ELT', ielt, elt)
            if isinstance(elt, tuple):
                elt = str(elt)
            #print(ielt, elt)
            worksheet.write(1 + ituple, 1 + ielt, elt)
       
        fname = t[cols.index(imcol)]
        img = imread_rgb(fname)
        img, _ = preprocess_face(img, None, False, 1, None, (100,100))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       
        is_success, buffer = cv2.imencode('.jpg', img)
        image_data = BytesIO(buffer)
        worksheet.insert_image(1 + ituple, 0, os.path.basename(fname), {"image_data": image_data})


    workbook.close()
    return None
