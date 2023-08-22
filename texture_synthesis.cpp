#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

int block_size;
int overlap_size;
int final_image_size;

void vertical_merge(vector<vector<vector<vector<uint8_t>>>> &all_blocks,vector<vector<vector<uint8_t>>> &final_image,int x,int y){
	int err=1e9;
	vector<vector<vector<uint8_t>>> final_arr(block_size,vector<vector<uint8_t>>(block_size,vector<uint8_t>(3,0)));
	for(int it=0;it<all_blocks.size();++it){
		int dp[block_size][overlap_size],cur_err=1e9,tot_err=0;
		for(int i=block_size-1;i>=0;i--){
			for(int j=0;j<overlap_size;j++){
				if(i==block_size-1){
					dp[i][j]=0;
					for(int k=0;k<3;k++) dp[i][j]+=pow((final_image[i+x][j+y][k]-all_blocks[it][i][j][k]),2);
				}
				else{
					dp[i][j]=dp[i+1][j];
					if(j!=0) dp[i][j]=min(dp[i][j],dp[i+1][j-1]);
					if(j!=overlap_size-1) dp[i][j]=min(dp[i][j],dp[i+1][j+1]);
					for(int k=0;k<3;k++) dp[i][j]+=pow((final_image[i+x][j+y][k]-all_blocks[it][i][j][k]),2);
				}
				for(int k=0;k<3;k++) tot_err+=pow((final_image[i+x][j+y][k]-all_blocks[it][i][j][k]),2);
			}
		}

		int vertical_cut[block_size];
		for(int i=0;i<overlap_size;i++){
			if(cur_err>dp[0][i]){
				cur_err=dp[0][i];vertical_cut[0]=i;
			}
		}

		for(int i=1;i<block_size;i++){
			int j=vertical_cut[i-1],ind=vertical_cut[i-1];
			int temp=dp[i][j];
			if(j!=0 && dp[i][j-1]<temp){
				ind=j-1;temp=dp[i][j-1];
			}
			if(j!=overlap_size-1 && dp[i][j+1]<temp) ind=j+1;
			vertical_cut[i]=ind;
		}

		if(tot_err<err){
			err=tot_err;
			for(int i=0;i<block_size;i++){
				for(int j=0;j<block_size;j++){
					for(int k=0;k<3;k++){
						if(vertical_cut[i]>j) final_arr[i][j][k]=final_image[i+x][j+y][k];
						else if(vertical_cut[i]==j) final_arr[i][j][k]=(final_image[i+x][j+y][k]+all_blocks[it][i][j][k])/2;
						else final_arr[i][j][k]=all_blocks[it][i][j][k];
					}
				}
			}
		}
	}
	for(int i=0;i<block_size;i++){
		for(int j=0;j<block_size;j++){
			for(int k=0;k<3;k++) final_image[i+x][j+y][k]=final_arr[i][j][k];
		}
	}
}

void horizontal_merge(vector<vector<vector<vector<uint8_t>>>> &all_blocks,vector<vector<vector<uint8_t>>> &final_image,int x,int y){
	int err=1e9;
	vector<vector<vector<uint8_t>>> final_arr(block_size,vector<vector<uint8_t>>(block_size,vector<uint8_t>(3,0)));
	for(int it=0;it<all_blocks.size();++it){
		int dp[overlap_size][block_size],cur_err=1e9,tot_err=0;
		for(int i=block_size-1;i>=0;i--){
			for(int j=0;j<overlap_size;j++){
				if(i==block_size-1){
					dp[j][i]=0;
					for(int k=0;k<3;k++) dp[j][i]+=pow((final_image[j+x][i+y][k]-all_blocks[it][j][i][k]),2);
				}
				else{
					dp[j][i]=dp[j][i+1];
					if(j!=0) dp[j][i]=min(dp[j][i],dp[j-1][i+1]);
					if(j!=overlap_size-1) dp[j][i]=min(dp[j][i],dp[j+1][i+1]);
					for(int k=0;k<3;k++) dp[j][i]+=pow((final_image[j+x][i+y][k]-all_blocks[it][j][i][k]),2);
				}
				for(int k=0;k<3;k++) tot_err+=pow((final_image[j+x][i+y][k]-all_blocks[it][j][i][k]),2);
			}
		}

		int horizontal_cut[block_size];
		for(int i=0;i<overlap_size;i++){
			if(cur_err>dp[i][0]){
				cur_err=dp[i][0];horizontal_cut[0]=i;
			}
		}
		for(int i=1;i<block_size-1;i++){
			int j=horizontal_cut[i-1],ind=horizontal_cut[i-1];
			int temp=dp[j][i];
			if(j!=0 && dp[j-1][i]<temp){
				ind=j-1;temp=dp[j-1][i];
			}
			if(j!=overlap_size-1 && dp[j+1][i]<temp) ind=j+1;
			horizontal_cut[i]=ind;
		}
		if(tot_err<err){
			err=tot_err;
			for(int j=0;j<block_size;j++){
				for(int i=0;i<block_size;i++){
					for(int k=0;k<3;k++){
						if(horizontal_cut[j]>i) final_arr[i][j][k]=final_image[i+x][j+y][k];
						else if(horizontal_cut[j]==i) final_arr[i][j][k]=(final_image[i+x][j+y][k]+all_blocks[it][i][j][k])/2;
						else final_arr[i][j][k]=all_blocks[it][i][j][k];
					}
				}
			}
		}
	}
	for(int i=0;i<block_size;i++){
		for(int j=0;j<block_size;j++){
			for(int k=0;k<3;k++) final_image[i+x][j+y][k]=final_arr[i][j][k];
		}
	}
}

void mixed_merge(vector<vector<vector<vector<uint8_t>>>> &all_blocks,vector<vector<vector<uint8_t>>> &final_image,int x,int y){
	int err=1e9;
	vector<vector<vector<uint8_t>>> final_arr(block_size,vector<vector<uint8_t>>(block_size,vector<uint8_t>(3,0)));
	for(int it=0;it<all_blocks.size();++it){
		int dp1[block_size][overlap_size],dp2[overlap_size][block_size-overlap_size],cur_err=1e9,cur_err1=1e9,tot_err=0;
		for(int i=block_size-1;i>=0;i--){
			for(int j=0;j<overlap_size;j++){
				if(i==block_size-1){
					dp1[i][j]=0;
					for(int k=0;k<3;k++) dp1[i][j]+=pow((final_image[i+x][j+y][k]-all_blocks[it][i][j][k]),2);
				}
				else{
					dp1[i][j]=dp1[i+1][j];
					if(j!=0) dp1[i][j]=min(dp1[i][j],dp1[i+1][j-1]);
					if(j!=overlap_size-1) dp1[i][j]=min(dp1[i][j],dp1[i+1][j+1]);
					for(int k=0;k<3;k++) dp1[i][j]+=pow((final_image[i+x][j+y][k]-all_blocks[it][i][j][k]),2);
				}
				for(int k=0;k<3;k++) tot_err+=pow((final_image[i+x][j+y][k]-all_blocks[it][i][j][k]),2);
			}
		}
		int vertical_cut[block_size];
		for(int i=0;i<overlap_size;i++){
			if(cur_err>dp1[0][i]){
				cur_err=dp1[0][i];vertical_cut[0]=i;
			}
		}
		for(int i=1;i<block_size;i++){
			int j=vertical_cut[i-1],ind=vertical_cut[i-1];
			int temp=dp1[i][j];
			if(j!=0 && dp1[i][j-1]<temp){
				ind=j-1;temp=dp1[i][j-1];
			}
			if(j!=overlap_size-1 && dp1[i][j+1]<temp) ind=j+1;
			vertical_cut[i]=ind;
		}

		for(int i=block_size-1;i>=overlap_size;i--){
			for(int j=0;j<overlap_size;j++){
				if(i==block_size-1){
					dp2[j][i-overlap_size]=0;
					for(int k=0;k<3;k++) dp2[j][i-overlap_size]+=pow((final_image[j+x][i+y][k]-all_blocks[it][j][i][k]),2);
				}
				else{
					dp2[j][i-overlap_size]=dp2[j][i+1-overlap_size];
					if(j!=0) dp2[j][i-overlap_size]=min(dp2[j][i-overlap_size],dp2[j-1][i+1-overlap_size]);
					if(j!=overlap_size-1) dp2[j][i-overlap_size]=min(dp2[j][i-overlap_size],dp2[j+1][i+1-overlap_size]);
					for(int k=0;k<3;k++) dp2[j][i-overlap_size]+=pow((final_image[j+x][i+y][k]-all_blocks[it][j][i][k]),2);
				}
				for(int k=0;k<3;k++) tot_err+=pow((final_image[j+x][i+y][k]-all_blocks[it][j][i][k]),2);
			}
		}

		int horizontal_cut[block_size-overlap_size];
		for(int i=0;i<overlap_size;i++){
			if(cur_err1>dp2[i][0]){
				cur_err1=dp2[i][0];horizontal_cut[0]=i;
			}
		}
		for(int i=1;i<block_size-1-overlap_size;i++){
			int j=horizontal_cut[i-1],ind=horizontal_cut[i-1];
			int temp=dp2[j][i];
			if(j!=0 && dp2[j-1][i]<temp){
				ind=j-1;temp=dp2[j-1][i];
			}
			if(j!=overlap_size-1 && dp2[j+1][i]<temp) ind=j+1;
			horizontal_cut[i]=ind;
		}

		if(tot_err<err){
			err=tot_err;
			for(int i=0;i<block_size;i++){
				for(int j=0;j<block_size;j++){
					for(int k=0;k<3;k++){
						if(vertical_cut[i]>j) final_arr[i][j][k]=final_image[i+x][j+y][k];
						else if(vertical_cut[i]==j) final_arr[i][j][k]=(final_image[i+x][j+y][k]+all_blocks[it][i][j][k])/2;
						else final_arr[i][j][k]=all_blocks[it][i][j][k];
					}
				}
			}
			for(int j=overlap_size;j<block_size;j++){
				for(int i=0;i<overlap_size;i++){
					for(int k=0;k<3;k++){
						if(horizontal_cut[j]>i) final_arr[i][j][k]=final_image[i+x][j+y][k];
						else if(horizontal_cut[j]==i) final_arr[i][j][k]=(final_image[i+x][j+y][k]+all_blocks[it][i][j][k])/2;
						else final_arr[i][j][k]=all_blocks[it][i][j][k];
					}
				}
			}
		}
	}
	for(int i=0;i<block_size;i++){
		for(int j=0;j<block_size;j++){
			for(int k=0;k<3;k++) final_image[i+x][j+y][k]=final_arr[i][j][k];
		}
	}
}

Mat image_creator(vector<vector<vector<uint8_t>>> vect,int r,int c,vector<vector<vector<vector<uint8_t>>>> &all_blocks){
	vector<vector<vector<uint8_t>>> farr(r,vector<vector<uint8_t>>(c,vector<uint8_t>(3)));

	int n=(final_image_size-overlap_size)/(block_size-overlap_size)+1;
	int sz=n*block_size-(n-1)*overlap_size;
	vector<vector<vector<uint8_t>>> final_image(sz,vector<vector<uint8_t>>(sz,vector<uint8_t>(3,0)));

	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			if(i==0 && j==0){
				int ind=rand()%all_blocks.size();
				for(int p=0;p<block_size;p++){
					for(int q=0;q<block_size;q++){
						for(int k=0;k<3;k++) final_image[p][q][k]=all_blocks[ind][p][q][k];
					}
				}
			}
			else if(i==0){
				vertical_merge(all_blocks,final_image,0,j*(block_size-overlap_size));
			}
			else if(j==0){
				horizontal_merge(all_blocks,final_image,i*(block_size-overlap_size),0);
			}
			else{
				mixed_merge(all_blocks,final_image,i*(block_size-overlap_size),j*(block_size-overlap_size));
			}
		}
	}


	Mat image(r, c, CV_8UC3);

	for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            Vec3b& pixel = image.at<cv::Vec3b>(i, j);
            for(int k=0;k<3;k++) pixel[k] = final_image[i][j][k];
        }
    }

	return image;
}

void generate_all_possible_blocks(vector<vector<vector<uint8_t>>> &initial_image_arr,vector<vector<vector<vector<uint8_t>>>> &all_blocks){
	int rsize=initial_image_arr.size(),csize=initial_image_arr[0].size();
	for(int i=0;i+block_size<rsize;i++){
		for(int j=0;j+block_size<csize;j++){
			vector<vector<vector<uint8_t>>> temp_block(block_size,vector<vector<uint8_t>>(block_size,vector<uint8_t>(3)));
			for(int r=0;r<block_size;r++){
				for(int c=0;c<block_size;c++){
					for(int k=0;k<3;k++) temp_block[r][c][k]=initial_image_arr[i+r][j+c][k];
				}
			}
			all_blocks.push_back(temp_block);
		}
	}
}

int main(int argc, char** argv)
{
	Mat image;
	image = imread(argv[1]);
	if (!image.data) {
		printf("No image data \n");
		return -1;
	}

	block_size=stoi(argv[2]);
	overlap_size=stoi(argv[3]);
	final_image_size=stoi(argv[4]);

	int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

	vector<vector<vector<uint8_t>>> initial_image_arr(rows,vector<vector<uint8_t>>(cols,vector<uint8_t>(channels)));
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			Vec3b pixel = image.at<cv::Vec3b>(i, j);
			for(int k=0;k<3;k++) initial_image_arr[i][j][k]=pixel[k];
		}
	}

	vector<vector<vector<vector<uint8_t>>>> all_blocks;
	generate_all_possible_blocks(initial_image_arr,all_blocks);

	Mat nimage=image_creator(initial_image_arr,final_image_size,final_image_size,all_blocks);
	
	imwrite("output.jpg", nimage);
	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", image);
	waitKey(0);
	imshow("Display Image", nimage);
	waitKey(0);
	return 0;
}

