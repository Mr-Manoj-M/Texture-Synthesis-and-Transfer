#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

int block_size;
int overlap_size;
int no_of_it;

void initial_block(vector<vector<vector<vector<uint8_t>>>> &all_blocks,vector<vector<vector<uint8_t>>> &Goal_tex_arr,vector<vector<vector<uint8_t>>> &final_image,vector<vector<vector<uint8_t>>> &prev_image,double alpha,double f){
	double err=1e14;
	if(f) f=1;

	for(int it=0;it<all_blocks.size();++it){
		double cur_err=0;
		for(int i=0;i<block_size;i++){
			for(int j=0;j<block_size;j++){
				for(int k=0;k<3;k++){
					cur_err+=alpha*f*pow((prev_image[i][j][k]-all_blocks[it][i][j][k]),2)+(1-alpha)*pow((Goal_tex_arr[i][j][k]-all_blocks[it][i][j][k]),2);
				}
			}
		}
		if(cur_err<err){
			err=cur_err;
			for(int i=0;i<block_size;i++){
				for(int j=0;j<block_size;j++){
					for(int k=0;k<3;k++) final_image[i][j][k]=all_blocks[it][i][j][k];
				}
			}
		}
	}
}

void vertical_merge(vector<vector<vector<vector<uint8_t>>>> &all_blocks,vector<vector<vector<uint8_t>>> &Goal_tex_arr,vector<vector<vector<uint8_t>>> &final_image,vector<vector<vector<uint8_t>>> &prev_image,int x,int y,double alpha,double f){
	double err=1e14;
	if(f) f=1;

	vector<vector<vector<uint8_t>>> final_arr(block_size,vector<vector<uint8_t>>(block_size,vector<uint8_t>(3,0)));
	for(int it=0;it<all_blocks.size();++it){
		double dp[block_size][overlap_size],cur_err=1e14;
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
		cur_err*=alpha;
		for(int i=x;i<x+block_size;i++){
			for(int j=y;j<y+block_size;j++){
				for(int k=0;k<3;k++){
					cur_err+=alpha*f*pow((prev_image[i][j][k]-all_blocks[it][i-x][j-y][k]),2)+(1-alpha)*pow((Goal_tex_arr[i][j][k]-all_blocks[it][i-x][j-y][k]),2);
				}
			}
		}

		if(cur_err<err){
			err=cur_err;
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

void horizontal_merge(vector<vector<vector<vector<uint8_t>>>> &all_blocks,vector<vector<vector<uint8_t>>> &Goal_tex_arr,vector<vector<vector<uint8_t>>> &final_image,vector<vector<vector<uint8_t>>> &prev_image,int x,int y,double alpha,double f){
	double err=1e14;
	if(f) f=1;

	vector<vector<vector<uint8_t>>> final_arr(block_size,vector<vector<uint8_t>>(block_size,vector<uint8_t>(3,0)));
	for(int it=0;it<all_blocks.size();++it){
		double dp[overlap_size][block_size],cur_err=1e14;
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

		cur_err*=alpha;
		for(int i=x;i<x+block_size;i++){
			for(int j=y;j<y+block_size;j++){
				for(int k=0;k<3;k++){
					cur_err+=alpha*f*pow((prev_image[i][j][k]-all_blocks[it][i-x][j-y][k]),2)+(1-alpha)*pow((Goal_tex_arr[i][j][k]-all_blocks[it][i-x][j-y][k]),2);
				}
			}
		}

		if(cur_err<err){
			err=cur_err;
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

void mixed_merge(vector<vector<vector<vector<uint8_t>>>> &all_blocks,vector<vector<vector<uint8_t>>> &Goal_tex_arr,vector<vector<vector<uint8_t>>> &final_image,vector<vector<vector<uint8_t>>> &prev_image,int x,int y,double alpha,double f){
	double err=1e14;
	if(f) f=1;

	vector<vector<vector<uint8_t>>> final_arr(block_size,vector<vector<uint8_t>>(block_size,vector<uint8_t>(3,0)));
	for(int it=0;it<all_blocks.size();++it){
		double dp1[block_size][overlap_size],dp2[overlap_size][block_size-overlap_size],cur_err=1e14,cur_err1=1e14;
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

		cur_err+=cur_err1;
		cur_err*=alpha;
		for(int i=x;i<x+block_size;i++){
			for(int j=y;j<y+block_size;j++){
				for(int k=0;k<3;k++){
					cur_err+=alpha*f*pow((prev_image[i][j][k]-all_blocks[it][i-x][j-y][k]),2)+(1-alpha)*pow((Goal_tex_arr[i][j][k]-all_blocks[it][i-x][j-y][k]),2);
				}
			}
		}

		if(cur_err<err){
			err=cur_err;
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

vector<vector<vector<uint8_t>>> image_creator(vector<vector<vector<uint8_t>>> &Goal_tex_arr,vector<vector<vector<uint8_t>>> &prev_image,vector<vector<vector<vector<uint8_t>>>> &all_blocks,int alpha,int f){
	int nrow=(prev_image.size()-overlap_size)/(block_size-overlap_size);
	int ncol=(prev_image[0].size()-overlap_size)/(block_size-overlap_size);
	int rsz=prev_image.size(),csz=prev_image[0].size();
	vector<vector<vector<uint8_t>>> final_image(rsz,vector<vector<uint8_t>>(csz,vector<uint8_t>(3,0)));
	for(int i=0;i<nrow;i++){cout<<i<<"i "<<endl;
		for(int j=0;j<ncol;j++){cout<<j<<"j "<<endl;
			if(i==0 && j==0){
				initial_block(all_blocks,Goal_tex_arr,final_image,prev_image,alpha,f);
			}
			else if(i==0){
				vertical_merge(all_blocks,Goal_tex_arr,final_image,prev_image,0,j*(block_size-overlap_size),alpha,f);
			}
			else if(j==0){
				horizontal_merge(all_blocks,Goal_tex_arr,final_image,prev_image,i*(block_size-overlap_size),0,alpha,f);
			}
			else{
				mixed_merge(all_blocks,Goal_tex_arr,final_image,prev_image,i*(block_size-overlap_size),j*(block_size-overlap_size),alpha,f);
			}
		}
	}

	return final_image;
}

void generate_all_possible_blocks(vector<vector<vector<uint8_t>>> &tex_image_arr,vector<vector<vector<vector<uint8_t>>>> &all_blocks){
	int rsize=tex_image_arr.size(),csize=tex_image_arr[0].size();
	for(int i=0;i+block_size<rsize;i++){
		for(int j=0;j+block_size<csize;j++){
			vector<vector<vector<uint8_t>>> temp_block(block_size,vector<vector<uint8_t>>(block_size,vector<uint8_t>(3)));
			for(int r=0;r<block_size;r++){
				for(int c=0;c<block_size;c++){
					for(int k=0;k<3;k++) temp_block[r][c][k]=tex_image_arr[i+r][j+c][k];
				}
			}
			all_blocks.push_back(temp_block);
		}
	}
}

int main(int argc, char** argv)
{
	Mat tex_image,Goal_tex;
	tex_image = imread(argv[1]);
	if (!tex_image.data) {
		printf("No Texture image data \n");
		return -1;
	}
	
	Goal_tex = imread(argv[2]);
	if (!Goal_tex.data) {
		printf("No Goal Texture image data \n");
		return -1;
	}
	block_size=stoi(argv[3]);
	overlap_size=stoi(argv[4]);
	no_of_it=stoi(argv[5]);

	int rows1 = tex_image.rows;
    int cols1 = tex_image.cols;
    int channels1 = tex_image.channels();
	vector<vector<vector<uint8_t>>> tex_image_arr(rows1,vector<vector<uint8_t>>(cols1,vector<uint8_t>(channels1)));

	for(int i=0;i<rows1;i++){
		for(int j=0;j<cols1;j++){
			Vec3b pixel = tex_image.at<cv::Vec3b>(i, j);
			for(int k=0;k<3;k++) tex_image_arr[i][j][k]=pixel[k];
		}
	}

	int rows2 = Goal_tex.rows;
    int cols2 = Goal_tex.cols;
    int channels2 = Goal_tex.channels();
	vector<vector<vector<uint8_t>>> Goal_tex_arr(rows2,vector<vector<uint8_t>>(cols2,vector<uint8_t>(channels2)));

	for(int i=0;i<rows2;i++){
		for(int j=0;j<cols2;j++){
			Vec3b pixel = Goal_tex.at<cv::Vec3b>(i, j);
			for(int k=0;k<3;k++) Goal_tex_arr[i][j][k]=pixel[k];
		}
	}

	vector<vector<vector<vector<uint8_t>>>> all_blocks;
	generate_all_possible_blocks(tex_image_arr,all_blocks);

	int nrow=(Goal_tex_arr.size()-overlap_size)/(block_size-overlap_size);
	int ncol=(Goal_tex_arr[0].size()-overlap_size)/(block_size-overlap_size);
	int rsz=nrow*block_size-(nrow-1)*overlap_size,csz=ncol*block_size-(ncol-1)*overlap_size;
	vector<vector<vector<uint8_t>>> prev_image(rsz,vector<vector<uint8_t>>(csz,vector<uint8_t>(3,0)));
	double alpha=0.1;
	vector<vector<vector<uint8_t>>> final_image(rsz,vector<vector<uint8_t>>(csz,vector<uint8_t>(3,0)));

	Mat nimage(rsz, csz, CV_8UC3);

	string s="output",s2=".png";

	for(int i=1;i<=no_of_it;i++){
		vector<vector<vector<uint8_t>>> ith_iteration_image=image_creator(Goal_tex_arr,prev_image,all_blocks,alpha,i-1);
		for(int i=0;i<rsz;i++){
			for(int j=0;j<csz;j++){
				for(int k=0;k<3;k++) final_image[i][j][k]=ith_iteration_image[i][j][k];
			}
		}
		if(no_of_it!=1) alpha=0.8*i/(no_of_it-1)+0.1;
		for (int i = 0; i < rsz; i++) {
        	for (int j = 0; j < csz; j++) {
            	Vec3b& pixel = nimage.at<cv::Vec3b>(i, j);
            	for(int k=0;k<3;k++) pixel[k] = final_image[i][j][k];
        	}
    	}
    	imwrite(s+s2, nimage);
    	s=s+'1';
	}
	
	namedWindow("Display Image", WINDOW_AUTOSIZE);
	imshow("Display Image", tex_image);
	waitKey(0);
	imshow("Display Image", Goal_tex);
	waitKey(0);
	imshow("Display Image", nimage);
	waitKey(0);
	return 0;
}

