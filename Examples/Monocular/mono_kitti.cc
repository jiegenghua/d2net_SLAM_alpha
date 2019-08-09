/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include"System.h"
#include "/usr/include/hdf5/serial/hdf5.h"
#include "/usr/include/hdf5/serial/H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    // 4 arguments: execute file argv[0]; vocabulary directory argv[1], 
    // setup path argv[2], image sequence path argv[3]
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    // load image; input: image path; output: vstrImageFilenames, vTimestamp
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);
    // how many images
    int nImages = vstrImageFilenames.size();
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    // calculate the tracking time
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    // read hdf5 file, one image and one keypoints and one descriptors
    string filename = "/home/mizhou/MyPro/ORB_SLAM2/kitti_hdf5.txt";
    // open the file
    ifstream hdf5file(filename);
    string str;
    vector<string> vecOfStrs;
    while(getline(hdf5file,str)){
       if(str.size()>0){
           vecOfStrs.push_back(str);
           }
    }
    hdf5file.close();

    for(int ni=18; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];
        cout<<"HDF:"<<ni<<"***********************"<<endl;
        const H5std_string FILE_NAME(vecOfStrs[ni]);
        const H5std_string DATASET1("keypoints");
        const H5std_string DATASET2("descriptors");
        // open the file
        H5File file(FILE_NAME,H5F_ACC_RDONLY);
        DataSet dataset1 = file.openDataSet(DATASET1);
        DataSet dataset2 = file.openDataSet(DATASET2); 
        DataSpace dataspace_key = dataset1.getSpace();
        DataSpace dataspace_des = dataset2.getSpace();
        int rank_key = dataspace_key.getSimpleExtentNdims();
        int rank_des = dataspace_des.getSimpleExtentNdims();
        hsize_t dims_key[2];
        hsize_t dims_des[2];
        int ndims_key = dataspace_key.getSimpleExtentDims(dims_key,NULL);
        int ndims_des = dataspace_des.getSimpleExtentDims(dims_des,NULL);

        //  this is for reading keypoint ********************************
        DataSpace mspace_key(rank_key,dims_key);
        float matrix_out_key[dims_key[0]][dims_key[1]];
        for (int i = 0; i < dims_key[0]; i++)
          for (int j = 0; j < dims_key[1]; j++)
            matrix_out_key[i][j] = 0;
        dataset1.read(matrix_out_key,PredType::NATIVE_FLOAT, mspace_key,dataspace_key);
    
        // this is for reading descriptors ****************************
        DataSpace mspace_des(rank_des,dims_des);
        float matrix_out_des[dims_des[0]][dims_des[1]];
        for (int i = 0; i < dims_des[0]; i++)
          for (int j = 0; j < dims_des[1]; j++)
            matrix_out_des[i][j] = 0;
        dataset2.read(matrix_out_des,PredType::NATIVE_FLOAT, mspace_des,dataspace_des);
        
        //  this is for giving keypoints to a mat
        cv::Mat keypoints_mat(dims_key[0],dims_key[1],CV_32F);
        for (int i = 0; i < dims_key[0]; i++){
          for (int j = 0; j < dims_key[1]; j++){
            keypoints_mat.at<float>(i,j) = matrix_out_key[i][j]*matrix_out_key[i][2];
          }
        }
        // this is for giving descriptors to a mat
        cv::Mat descriptors_mat(dims_des[0],dims_des[1],CV_8U);
        for (int i = 0; i < dims_des[0]; i++){
          for (int j = 0; j < dims_des[1]; j++){
            descriptors_mat.at<uchar>(i,j) = floor(255*matrix_out_des[i][j]);
            }
        }
        //cout<<descriptors_mat.at<float>(0,1);
        
        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        // if the compiler can compile c++11
        // get the current time
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe,keypoints_mat,descriptors_mat);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        // calculate the time for trakcing this image
        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    // after trakcing all the images
    SLAM.Shutdown();

    // Tracking time statistics
    // calculate the median, total, average
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    // timestamp for reading image
    string strPathTimeFile = strPathToSequence + "/times.txt";
    // string to char*
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    // use the file in image_0, which is the left camera sequence for double cameras
    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        // set the path of 0-nTime-1 images
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}
