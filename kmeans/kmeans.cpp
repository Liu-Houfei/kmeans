#include <iostream>
#include <cstdlib> 
#include <math.h>
#include <time.h> 
#include <fstream>
#include<string>
#include<sstream>
#include <iomanip>

using namespace std;

//最大值
#define DBL_MAX  1.7e+308  
//最小值
#define DBL_MIN -1.7e+308
//算法名称
#define algorithmName  "MyKmeans"
//执行kmeans的次数
#define MAX_RUNS 50
//最大迭代次数
#define iterationMax 1000
// 中心初始化方法
//0--前clu_num个数;
//1--手工指定;
//2--随机选择
#define initialCase 1

double** initDoubleArray(int row, int column);
double ** initDoubleArray(int row, int column, double defaultVal);
int ** initIntArray(int row, int column);
int ** initIntArray(int row, int column, int defaultVal);
double** getDoubleDataFromTxt(string path, int row, int column);
int** getIntDataFromTxt(string path, int row, int column);
void printArray(double** data, int row, int column);
void printArray(int** data, int row, int column);
char* getCurTimeChar();
int getCharArrayCount(char charName[], int size);
double** initCenters(int initCase, double **simpleData, double **clusterData, int **labelData, int row, int dataDim, int clusterNum);
double** initClusterData_1(double **clusterData, double  **simpleData, int **labelData, int row, int dataDim, int clusterNum);

void kmeans(double **simpleData, double **clusterData, int **labelData,
	int *mark, int maxRuns, int initCase, int &iterTimes, double &distTimesAll, bool &isHaveEmptyCluster,
	int dataNum, int dataDim, int clusterNum, int runTimes);

double getDistFromIToJ(double **d1, double **d2, int di, int dj, int dataNum, int clusterNum, int dataDim);
bool updateCenters(double **simpleData, double **clusterData, double **clusterDataTmp, int *mark, int *numClu, int dataNum, int clusterNum, int dataDim);
bool isCenterEquals(double **clusterData, double **clusterDataTmp, int clusterNum, int dataDim);
bool isCenterEquals(int *arr1, int *arr2, int n);
void writeHeader(string path, string currentTime);
void writeContent(string path, int runTimes, int cluNum, int dataNum, int dataDim,
	int iterTimes, double objFunc, double distCallsAll, double timeAll, double **resultMatAll);
void writeFooter(string path);
void writeMeanResult(string path, double **resultMatAll, int maxRuns, int resultCols);
double calObjFuncForGivenCents(double **data, double **V, int data_num, int data_dim, int clu_num);
void copy(double**clusterData, double** clusterDataTmp, int clusterNum, int dataDim);
void release(double **data, int cow, int col);


void main()
{
	cout << "输入算法参数：" << endl << endl << endl << "参数输入注意事项" << endl;
	cout << "数据字符串不包括扩展名" << endl;
	cout << "文件名称不区分大小写" << endl;
	cout << endl << endl;
	//输入txt数据文件名
	cout << "1、输入 数据名称：" << endl << " dataName = ";
	string dataName = "iris(150-4)";
	//cin >> dataName;
	//输入算法运行次数
	cout << "2、输入算法运行次数" << endl << "注意：maxRuns 必须小于等于50" << endl << " maxRuns = ";
	int maxRuns = 50;
	//cin >> maxRuns;
	//输入数据个数
	cout << "3、输入数据个数：" << endl << " N = ";
	int dataNum = 150;
	//cin >> dataNum;
	//维度
	cout << "4、输入数据维数d：" << endl << " d = ";
	int dataDim = 4;
	//cin >> dataDim;
	//聚类个数
	cout << "5、输入 聚类个数C：" << endl << " C = ";
	int clusterNum = 5;
	//cin >> clusterNum;

	//创建存储样本点数据的二维数组simpleDate，大小=数据个数*维度
	double **simpleData = initDoubleArray(dataNum, dataDim);
	//创建存储标签数据的二维数组，大小=算法运行次数*聚类个数
	int **labelData = initIntArray(maxRuns, clusterNum);
	//创建存储聚类中心的二维数组，大小=聚类个数*维度
	double **clusterData = initDoubleArray(clusterNum, dataDim);
	//创建临时存储聚类中心的二维数组，大小=聚类个数*维度
	double **clusterDataTmp = initDoubleArray(clusterNum, dataDim);
	//创建存储运行结果的二维数组，大小=运行次数*10
	double** resultFile = initDoubleArray(maxRuns, 10);

	//样本点文件磁盘目录
	string simpleDateDrirectory = "F:\\kmeans\\Fast KMeans Tested Data Sets\\";
	//便签文件磁盘目录
	string labelDateDirectory = "F:\\kmeans\\initIDXForCenters\\";
	//结果存放目录
	string resultPath = "F:\\kmeans\\";
	//文件后缀名
	string endWith = ".txt";
	//simpleDataName 样本文件名
	stringstream simpleDataName;
	simpleDataName << simpleDateDrirectory << dataName << endWith;
	//从磁盘读取数据文件
	simpleData = getDoubleDataFromTxt(simpleDataName.str(), dataNum, dataDim);
	//printArray(simpleData, dataNum, dataDim); //输出样本点数据测试通过

	//组合【样本点个数N+聚类个数C+运行次数maxRuns】形成新字符串，该字符串是标签文件名
	//labelDataName 标签文件名
	stringstream labelDataName;
	labelDataName << labelDateDirectory << "N=" << dataNum
		<< "_c=" << clusterNum << "_Runs=" << maxRuns << endWith;
	//从磁盘读取标签文件
	labelData = getIntDataFromTxt(labelDataName.str(), maxRuns, clusterNum);
	//printArray(initCenterIDXMat, maxRuns, clusterNum); //输出标签数据测试通过

	//构建存储文件名称
	string currentTime = getCurTimeChar();
	stringstream fileName;
	fileName << resultPath << dataName
		<< "_c=" << clusterNum << "_dim=" << dataDim << "_dataNum=" << dataNum << "_maxRunTimes=" << maxRuns
		<< "_" << algorithmName
		<< "_" << currentTime << endWith;
	cout << endl << fileName.str() << endl;  //存储文件名称测试通过


	//开始循环，循环次数<maxRuns
	for (int runTimes = 1; runTimes <= MAX_RUNS; runTimes++) {
		//mark数组存储样本点的标记（样本点属于哪个簇）
		int *mark = new int[dataNum];
		//kmeans最大迭代次数
		int iterMax = iterationMax;
		//初始化中心的方式
		int initCase = initialCase;
		//迭代的次数，在标准k-means迭代次数，决定这k-means的运行时间，迭代次数越少，本次k-means运行时间越少
		int iterTimes = 0;
		//计算距离的总次数
		double distTimesAll = 0;
		//是否存在空簇
		bool ifHaveEmpt = false;
		//创建定时器，记录运行时时间
		clock_t T_start, T_end;
		T_start = clock();
		kmeans(simpleData, clusterData, labelData, mark,
			iterMax, initCase, iterTimes, distTimesAll, ifHaveEmpt,
			dataNum, dataDim, clusterNum, runTimes);
		T_end = clock();
		//运行时间差
		double timeAll = T_end - T_start;
		double objFunc = calObjFuncForGivenCents(simpleData, clusterData, dataNum, dataDim, clusterNum);
		//写入头
		if (runTimes == 1) {
			writeHeader(fileName.str(), currentTime);
		}
		//写入数据
		writeContent(fileName.str(), runTimes, clusterNum, dataNum, dataDim, iterTimes, objFunc,distTimesAll, timeAll, resultFile);
	}
	//写入尾
	writeFooter(fileName.str());
	//写入均值
	writeMeanResult(fileName.str(), resultFile, maxRuns, 10);
}

void writeHeader(string path, string createTime) {
	ofstream outfile;
	//追加模式
	outfile.open(path, ios::app);
	outfile << "created by liuhoufei |" << createTime << endl;
	outfile << setw(28) << setiosflags(ios::left) << "runTimes";
	outfile << setw(28) << setiosflags(ios::left) << "cluNum";
	outfile << setw(28) << setiosflags(ios::left) << "dataNum";
	outfile << setw(28) << setiosflags(ios::left) << "dataDim";
	outfile << setw(28) << setiosflags(ios::left) << "iterTimes";
	outfile << setw(28) << setiosflags(ios::left) << "objFunc";
	outfile << setw(28) << setiosflags(ios::left) << "distCallsAll";
	outfile << setw(28) << setiosflags(ios::left) << "distCallPerIterPerData";
	outfile << setw(28) << setiosflags(ios::left) << "timeAll(毫秒)";
	outfile << setw(28) << setiosflags(ios::left) << "timePerIter(毫秒)";
	outfile.close();
}
void writeContent(string path, int runTimes, int cluNum, int dataNum, int dataDim,
	int iterTimes, double objFunc, double distCallsAll, double timeAll, double **resultMatAll) {
	int rows = runTimes - 1;
	int cols = 0;
	double distCallPerIterPerData = double(distCallsAll) / double(iterTimes) / double(dataNum);
	double timePerIter = timeAll / double(iterTimes);
	ofstream outfile;
	outfile.open(path, ios::app);
	outfile << endl;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << runTimes;		resultMatAll[rows][cols] = runTimes; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << cluNum;		resultMatAll[rows][cols] = cluNum; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << dataNum;		resultMatAll[rows][cols] = dataNum; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << dataDim;		resultMatAll[rows][cols] = dataDim; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << iterTimes;		resultMatAll[rows][cols] = iterTimes; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << objFunc;		resultMatAll[rows][cols] = objFunc; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << distCallsAll;	resultMatAll[rows][cols] = distCallsAll; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << distCallPerIterPerData;	resultMatAll[rows][cols] = distCallPerIterPerData; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << timeAll;			resultMatAll[rows][cols] = timeAll; cols++;
	outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << timePerIter;	resultMatAll[rows][cols] = timePerIter; cols++;
	outfile.close();
}
void writeFooter(string path) {
	ofstream outfile;
	outfile.open(path, ios::app);
	outfile << endl;
	outfile << endl;
	outfile << setw(28) << setiosflags(ios::left) << "runTimes";
	outfile << setw(28) << setiosflags(ios::left) << "cluNum";
	outfile << setw(28) << setiosflags(ios::left) << "dataNum";
	outfile << setw(28) << setiosflags(ios::left) << "dataDim";
	outfile << setw(28) << setiosflags(ios::left) << "iterTimes";
	outfile << setw(28) << setiosflags(ios::left) << "objFunc";
	outfile << setw(28) << setiosflags(ios::left) << "distCallsAll";
	outfile << setw(28) << setiosflags(ios::left) << "distCallPerIterPerData";
	outfile << setw(28) << setiosflags(ios::left) << "timeAll(毫秒)";
	outfile << setw(28) << setiosflags(ios::left) << "timePerIter(毫秒)";
	outfile.close();
}

void writeMeanResult(string path, double **resultMatAll, int maxRuns, int resultCols) {
	double *resultVecMean = new double[resultCols];

	for (int l = 0; l < resultCols; l++) {
		resultVecMean[l] = 0;
	}
	for (int col = 0; col < resultCols; col++)
	{
		double tmp = 0;
		for (int row = 0; row < maxRuns; row++)
		{
			tmp = tmp + resultMatAll[row][col];
		}
		resultVecMean[col] = tmp / double(maxRuns);
	}
	ofstream outfile;
	outfile.open(path, ios::app);
	outfile << endl;
	for (int col = 0; col < resultCols; col++)
	{
		outfile << setw(28) << setprecision(20) << setiosflags(ios::left) << resultVecMean[col];
	}
	outfile.close();
}

/*
根据指定值初始化int型二维数组（默认值=0）
row:行
column:列
*/
int ** initIntArray(int row, int column) {
	int i, j;
	//给二维数组分配内存
	int **array = new int*[row];
	for (i = 0; i < row; i++) {
		array[i] = new int[column];
	}
	//给二维数组赋指定值
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			array[i][j] = 0;
		}
	}
	return array;
}
/*
根据指定值初始化int型二维数组
row:行
column:列
defaultVal:设置默认值
*/
int ** initIntArray(int row, int column, int defaultVal) {
	int i, j;
	//给二维数组分配内存
	int **array = new int*[row];
	for (i = 0; i < row; i++) {
		array[i] = new int[column];
	}
	//给二维数组赋指定值
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			array[i][j] = defaultVal;
		}
	}
	return array;
}
/*
释放二维数组空间（有问题，释放不了内存）
*/
void release(double **data, int row, int col) {
	for (int i = 0; i < row; i++)
	{
		delete[] data[i];
	}
	delete[] data;
}
/*
根据指定值初始化double型二维数组（默认值=0.0）
row:行
column:列
*/
double ** initDoubleArray(int row, int column) {
	int i, j;
	//给二维数组分配内存
	double **array = new double*[row];
	for (i = 0; i < row; i++) {
		array[i] = new double[column];
	}
	//给二维数组赋指定值
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			array[i][j] = 0.0;
		}
	}
	return array;
}
/*
根据指定值初始化double型二维数组
row:行
column:列
defaultVal:设置默认值
*/
double ** initDoubleArray(int row, int column, double defaultVal) {
	int i, j;
	//给二维数组分配内存
	double **array = new double*[row];
	for (i = 0; i < row; i++) {
		array[i] = new double[column];
	}
	//给二维数组赋指定值
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			array[i][j] = defaultVal;
		}
	}
	return array;
}
/*
从txt文本中读取double数据
path:文件路径
row:行
column:列
*/
double** getDoubleDataFromTxt(string path, int row, int column) {
	//动态创建二维数组
	double **data = initDoubleArray(row, column);
	//定义文件读入流
	ifstream in;
	//打开指定路径文件
	in.open(path);
	int i, j;
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			in >> data[i][j];
		}
	}
	//关闭文件流
	in.close();
	//返回从文件中读取的用二维数组存储的数据
	return data;
}
/*
从txt文本中读取int数据
path:文件路径
row:行
column:列
*/
int** getIntDataFromTxt(string path, int row, int column) {
	//动态创建二维数组
	int **data = initIntArray(row, column);
	//定义文件读入流
	ifstream in;
	//打开指定路径文件
	in.open(path);
	int i, j;
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			in >> data[i][j];
		}
	}
	//关闭文件流
	in.close();
	//返回从文件中读取的用二维数组存储的数据
	return data;
}
/*
打印double型数组信息
data:二维数组
row:行
column:列
*/
void printArray(double** data, int row, int column) {
	int i, j;
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			cout << data[i][j] << " ";
		}
		cout << endl;
	}
}
/*
打印int型数组信息
data:二维数组
row:行
column:列
*/
void printArray(int** data, int row, int column) {
	int i, j;
	for (i = 0; i < row; i++) {
		for (j = 0; j < column; j++) {
			cout << data[i][j] << " ";
		}
		cout << endl;
	}
}
// 返回当前系统的时间字符串，自己去掉了一些冒号等无法取为文件名称的字符，替换成好看的能命名文件的字符串
// 注意：动态new开辟的指针，开辟大小依据空格分界前的元素来定，最后一个元素是空格，即对应的整型是 0 ；
char* getCurTimeChar()
{
	time_t t = time(0);
	char CurTimeChar[64];
	strftime(CurTimeChar, sizeof(CurTimeChar), "%Y-%m-%d-%X", localtime(&t));
	int count = getCharArrayCount(CurTimeChar, 64);
	char s[2] = ":";
	for (int i = 0; i < count; i++)
	{
		if (CurTimeChar[i] == s[0])
		{
			CurTimeChar[i] = 45;
		}
	}
	char * char_ptr = new char[count];
	for (int i = 0; i < (count); i++)
	{
		char_ptr[i] = CurTimeChar[i];

	}
	return char_ptr;
}
// 计算当前 char型数组的在空格以前的元素个数，以空格为界，这个在定义过大的char型数组时候有用，因为数组定义过大以后，最后一位是空格标记,注意包括空格，空格必须包括计数，因为char最后一个元素必须有空格
// 适用范围： 1、 定义的char数组过大，如char s[100] = "sfs";此时用这个函数会返回4(因为包括空格标记)；2、本身不含有空格，类似这个例子，如果改为char s[100] = "s fs";则返回的就是1了,错误了
int getCharArrayCount(char charName[], int size)
{
	int tmpCount = 0;
	for (int i = 0; i < size; i++)
	{
		if (charName[i] != 0)
		{
			tmpCount++;
		}
		else
		{
			tmpCount++; break;
		}
	}
	return tmpCount;
}
/*
kmeans聚类算法
simpleData：样本点数据二维数组，大小=样本点个数dataNum * 维度dataDim
clusterData：聚类中心二维数组，大小=聚类个数clusterNum* 维度dataDim
clusterDataTmp：临时聚类中心二维数组，大小=聚类个数clusterNum* 维度dataDim
labelData：标签二维数组
ind：记录每个样本所属的簇
mexItea：最大迭代次数
initCase：初始化中心点方式
iterTime：迭代次数
distcallAlls：距离计算总次数
ifHaveEmpt：是否包含空簇
dataNum：样本点个数
dataDim：样本维度
clusterNum：聚类中心个数

*/
void kmeans(double **simpleData, double **clusterData, int **labelData,
	int *mark, int maxRuns, int initCase, int &iterTimes, double &distTimesAll, bool &isHaveEmptyCluster,
	int dataNum, int dataDim, int clusterNum, int runTimes) {
	//记录整个过程的计算距离次数
	distTimesAll = 0;
	//记录1次迭代的计算距离次数
	double distTimes;
	//记录是否有空簇
	isHaveEmptyCluster = false;
	//最小距离，计算距离
	double minDis = 0, distIJ = 0;
	//每个聚类的样本点个数
	int *numClu = new int[clusterNum];
	int *markTmp = new int[dataNum];
	double** clusterDataTmp = initDoubleArray(clusterNum, dataDim);
	//根据初始化方式initCase初始化中心点
	clusterData = initCenters(initCase, simpleData, clusterData, labelData, runTimes, dataDim, clusterNum);
	//cout << endl; printArray(clusterData, clusterNum, dataDim);  //输出初始中心点测试成功
	//开始迭代maxRuns = 1000
	for (iterTimes = 1; iterTimes <= maxRuns; iterTimes++) {
		distTimes = 0;
		for (int i = 0; i < dataNum; i++) {
			minDis = DBL_MAX;
			for (int j = 0; j < clusterNum; j++) {
				distIJ = getDistFromIToJ(simpleData, clusterData, i, j, dataNum, clusterNum, dataDim);
				if (distIJ < minDis) {
					minDis = distIJ;
					//设置样本点的所属簇
					mark[i] = j;
				}
				//每计算一次距离+1
				distTimes++;
			}
		}//1次迭代结束
		//将每次迭代计算距离的次数加和
		distTimesAll += distTimes;
		//判断新旧两个聚类是否相等
		//if (isCenterEquals(clusterData, clusterDataTmp, clusterNum, dataDim)) {
		//	return;
		//}
		if (isCenterEquals(mark, markTmp, dataNum)) {
			return;
		}
		else {
			//更新中心点
		    //创建临时clusterDataTmp存储旧的clusterData
			isHaveEmptyCluster = updateCenters(simpleData, clusterData, clusterDataTmp,
				mark, numClu, dataNum, clusterNum, dataDim);
			//cout << "clusterData:" << endl;
			//printArray(clusterData, clusterNum, dataDim);
			//cout << "clusterDataTmp:" << endl;
			//printArray(clusterDataTmp, clusterNum, dataDim);
			//将新中心替换旧中心
			copy(clusterData, clusterDataTmp, clusterNum, dataDim);
			for (int i = 0; i < dataNum; i++) {
				markTmp[i] = mark[i];
			}
		}
	}
	return;
}

/*
计算两点之间的距离
*/
double getDistFromIToJ(double **d1, double **d2, int di, int dj, int dataNum, int clusterNum, int dataDim) {
	double dist2;
	dist2 = 0;
	for (int l = 0; l < dataDim; l++)
	{
		dist2 += (d1[di][l] - d2[dj][l])*(d1[di][l] - d2[dj][l]);
	}
	return dist2;
}
/*
初始化聚类中心点

row使用标签二维数组指定行的数据
*/
double** initCenters(int initCase, double **simpleData, double **clusterData, int **labelData, int row, int dataDim, int clusterNum) {
	if (initCase == 0) {

	}
	if (initCase == 1) {
		return initClusterData_1(clusterData, simpleData, labelData, row, dataDim, clusterNum);
	}
	if (initCase == 2) {

	}
	return NULL;
}

/*
手工指定初始化聚类中心
clusterData：聚类中心二维数组
simpleData：样本点二维数组
labelData：标签二维数组
row：使用标签二维数组指定行的数据
clusterNum：聚类个数
*/
double** initClusterData_1(double **clusterData, double  **simpleData, int **labelData, int row, int dataDim, int clusterNum) {
	for (int i = 0; i < clusterNum; i++) {
		int index = labelData[row - 1][i];
		for (int j = 0; j < dataDim; j++) {
			clusterData[i][j] = simpleData[index][j];
		}
	}
	return clusterData;

}
//clusterData = updateCenters(simpleData,clusterData,clusterDataTmp,mark,numClu,dataNum,clusterNum,dataDim);
/*
更新聚类中心并判断是否有空簇
numClu：记录每个簇的样本点数
*/
bool updateCenters(double **simpleData, double **clusterData, double **clusterDataTmp,
	int *mark, int *numClu, int dataNum, int clusterNum, int dataDim) {
	bool isHaveClusterEmpty = false;
	//初始化簇
	for (int i = 0; i < clusterNum; i++) {
		numClu[i] = 0;
		for (int j = 0; j < dataDim; j++) {
			clusterDataTmp[i][j] = 0;
		}
	}
	for (int i = 0; i < dataNum; i++) {
		//index为样本点i所属的簇下标
		int index = mark[i];
		for (int j = 0; j < dataDim; j++) {
			//将簇中每个样本点的每个维度相加
			clusterDataTmp[index][j] += simpleData[i][j];
		}
		//记录每个簇的样本点数
		numClu[index]++;
	}
	for (int i = 0; i < clusterNum; i++) {
			for (int j = 0; j < dataDim; j++) {
				if (numClu[i] != 0) {  //不是空簇
					clusterDataTmp[i][j] = clusterDataTmp[i][j] / numClu[i];
				}
				else {
					clusterDataTmp[i][j] = clusterData[i][j];
					isHaveClusterEmpty = true;
				}
			}
	}
	return isHaveClusterEmpty;
}

/*
判断2个二维数组是否相等
*/
bool isCenterEquals(double **clusterData, double **clusterDataTmp, int clusterNum, int dataDim) {
	for (int i = 0; i < clusterNum; i++) {
		for (int j = 0; j < dataDim; j++) {
			if (!(abs(clusterData[i][j] - clusterDataTmp[i][j]) < DBL_EPSILON)) {
				return false;
			}
		}
	}
	return true;
}
/*
判断两个一维数组是否相等
*/
bool isCenterEquals(int *arr1, int *arr2, int n) {
	for (int i = 0; i < n; i++) {
		if (arr1[i] != arr2[i]) {
			return false;
		}
	}
	return true;
}

//  通过给定的中心与数据，计算目标函数值
double calObjFuncForGivenCents(double **data, double **V, int data_num, int data_dim, int clu_num)
{
	double objFunc = 0;
	double tmpMinDistSquare=0; // 距离平方
	double tmpDistSquare=0;  // 距离平方
	for (int i = 0; i < data_num; i++)
	{
		tmpMinDistSquare = DBL_MAX;
		for (int j = 0; j < clu_num; j++)
		{
			tmpDistSquare = 0;
			for (int l = 0; l < data_dim; l++)
			{
				tmpDistSquare += ((data[i][l] - V[j][l])*(data[i][l] - V[j][l]));
			}
			if (tmpDistSquare < tmpMinDistSquare)
			{
				tmpMinDistSquare = tmpDistSquare;
			}	
		}
		objFunc += tmpMinDistSquare;
	}
	return objFunc;
}

/*

*/

void copy(double**clusterData, double** clusterDataTmp, int clusterNum, int dataDim) {
	for (int i = 0; i < clusterNum; i++) {
		for (int j = 0; j < dataDim; j++) {
			clusterData[i][j] = clusterDataTmp[i][j];
		}
	}
}

