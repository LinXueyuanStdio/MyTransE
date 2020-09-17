#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <set>
#include <pthread.h>
#include <vector>
#include <map>
#include <iostream>

using namespace std;

const float pi = 3.141592653589793238462643383;

int transeThreads = 8;
int transeTrainTimes = 3000;
int nbatches = 1;
int dimension = 200;
float transeAlpha = 0.001;
float margin = 1;
int L1_flag = 1;
double combination_threshold = 3;
int combination_restriction = 5000;
int entityTotal;
int attributeTotal,valueTotal,aTripleTotal;
int a_transeLen;
int a_transeBatch;
float res;
string lang = "fr_en";

unsigned long long * next_random;
int *aLefHead, *aRigHead;
int *aLefTail, *aRigTail;
float *entityVec;
float *attributeVec,*valueVec;

set<int> commonEntities;
vector<int> entitiesInKg1, entitiesInKg2;
map<int, int> correspondingEntity;
vector<float> combinationProbability;

string inPath = "data/fr_en/";
string outPath = "res/";

void out_transe(string);

struct Triple 
{
	int h, r, t;
};

Triple *aTrainHead, *aTrainTail, *aTrainList;

struct cmp_head 
{
	bool operator()(const Triple &a, const Triple &b) 
	{
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail 
{
	bool operator()(const Triple &a, const Triple &b) 
	{
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

unsigned long long randd(int id) 
{ 
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) 
{ 
	int res = randd(id) % x; 
	while (res<0)
		res+=x;
	return res;
}

float rand(float min, float max) 
{
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) 
{ 
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float sigmoid(float x)
{ 
	return 1.0/(1.0 + exp(-x));
}

float randn(float miu,float sigma, float min ,float max) 
{
	float x, y, dScope;
	do {
	    x = rand(min,max); 
	    y = normal(x,miu,sigma); 
	    dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

void norm(float * con) 
{
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
	    x += (*(con + ii)) * (*(con + ii)); 
	x = sqrt(x);
	if (x>1) 
		for (int ii=0; ii < dimension; ii++)
		    *(con + ii) /= x; 
}

void init() 
{ 
	FILE *fin;
	int tmp;

    cout<<"0"<<endl;
    //属性向量初始化
    tmp = 1;
    attributeTotal=10969;
    attributeVec = (float *)calloc(attributeTotal * dimension, sizeof(float));
    for (int i = 0; i < attributeTotal; i++) 
    {
        for (int ii=0; ii<dimension; ii++)
            attributeVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
    }

    //实体向量初始化
//	tmp = 1;
//	entityTotal = 24902;
//	entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
//	for (int i = 0; i < entityTotal; i++)
//	{
//		for (int ii=0; ii<dimension; ii++)
//		    entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
//		norm(entityVec+i*dimension);
//	}
    //实体向量初始化
    fin = fopen((inPath+"transe_init.txt").c_str(), "r");
    entityTotal = 39654;
    cout<<"1"<<endl;
    entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
    for(int i=0; i<entityTotal*dimension; i++){
        tmp = fscanf(fin, "%f", &entityVec[i]);
    }

    cout<<"2"<<endl;

    //属性值向量初始化
    tmp = 1;
    valueTotal = 410207;
    valueVec = (float *)calloc(valueTotal * dimension, sizeof(float));
    for (int i = 0; i < valueTotal; i++) 
    {
        for (int ii=0; ii<dimension; ii++)
            valueVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
        norm(valueVec+i*dimension);
    }
    cout<<"3"<<endl;

    combinationProbability.resize(entityTotal);
    fill(combinationProbability.begin(), combinationProbability.end(), 0); 

    fin = fopen((inPath + "att_triple_all").c_str(), "r");
    aTripleTotal = 1105208;
//    tmp = fscanf(fin, "%d", &aTripleTotal);  //属性三元组的总数量
    aTrainHead = (Triple *)calloc(aTripleTotal, sizeof(Triple));
    aTrainTail = (Triple *)calloc(aTripleTotal, sizeof(Triple));
    aTrainList = (Triple *)calloc(aTripleTotal, sizeof(Triple));
    aTripleTotal = 0;
    cout<<"4"<<endl;
    while (fscanf(fin, "%d", &aTrainList[aTripleTotal].h) == 1)
    {
        tmp = fscanf(fin, "%d", &aTrainList[aTripleTotal].t);
        tmp = fscanf(fin, "%d", &aTrainList[aTripleTotal].r); //读入属性三元组的 h a v
        aTrainHead[aTripleTotal].h = aTrainList[aTripleTotal].h;
        aTrainHead[aTripleTotal].t = aTrainList[aTripleTotal].t;
        aTrainHead[aTripleTotal].r = aTrainList[aTripleTotal].r;
        aTrainTail[aTripleTotal].h = aTrainList[aTripleTotal].h;
        aTrainTail[aTripleTotal].t = aTrainList[aTripleTotal].t;
        aTrainTail[aTripleTotal].r = aTrainList[aTripleTotal].r;
        aTripleTotal++;
    }
    fclose(fin);
    sort(aTrainHead, aTrainHead + aTripleTotal, cmp_head());
    sort(aTrainTail, aTrainTail + aTripleTotal, cmp_tail());
    cout<<"5"<<endl;

    aLefHead = (int *)calloc(valueTotal, sizeof(int));
    aRigHead = (int *)calloc(valueTotal, sizeof(int));
    aLefTail = (int *)calloc(valueTotal, sizeof(int));
    aRigTail = (int *)calloc(valueTotal, sizeof(int));
    memset(aRigHead, -1, sizeof(aRigHead));
    memset(aRigTail, -1, sizeof(aRigTail));
    for (int i = 1; i < aTripleTotal; i++) 
    {
        if (aTrainTail[i].t != aTrainTail[i - 1].t) 
        {
            aRigTail[aTrainTail[i - 1].t] = i - 1;
            aLefTail[aTrainTail[i].t] = i;
        }
        if (aTrainHead[i].h != aTrainHead[i - 1].h) 
        { 
            aRigHead[aTrainHead[i - 1].h] = i - 1;
            aLefHead[aTrainHead[i].h] = i;
        }
    }
    aRigHead[aTrainHead[aTripleTotal - 1].h] = aTripleTotal - 1;
    aRigTail[aTrainTail[aTripleTotal - 1].t] = aTripleTotal - 1;
    cout<<"6"<<endl;

    //载入已知的对齐实体 没用到
    int commonTotal; 
    fin = fopen((inPath + "ref_ent_ids").c_str(), "r");
    tmp = fscanf(fin, "%d", &commonTotal);

    for(int i = 0;i<commonTotal;i++)
    { 
    	int entId;
    	tmp = fscanf(fin, "%d", &entId);
    	commonEntities.insert(entId); 
    }
    printf("%d known entities pairs.\n", commonTotal);

    //载入KG1和KG2的实体？为什么
    for(int i = 0;i<entityTotal;i++)
    {

    	if(!commonEntities.count(i)) //没出现在种子实体对中
    	{ 
    	    if(i < 14951) 
    	    { 
    	        entitiesInKg1.push_back(i); 
    	    }
    	    else 
    	    	entitiesInKg2.push_back(i);
    	}
    }
    fclose(fin);
}

float calc_sum(int e1, int e2, int rel)  //计算v-h-a
{
	float sum=0;
	int last1 = e1 * dimension; //实体
	int last2 = e2 * dimension; //属性值
	int lastr = rel * dimension; //属性
    for (int ii=0; ii < dimension; ii++) 
    {
        sum += fabs(valueVec[last2 + ii] - entityVec[last1 + ii] - attributeVec[lastr + ii]); 
    }
	return sum;
}


void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) 
{

	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimension;
	int lastb1 = e1_b * dimension;
	int lastb2 = e2_b * dimension;
	int lastbr = rel_b * dimension;
    for (int ii=0; ii  < dimension; ii++) 
    {
        float x;
        x = (valueVec[lasta2 + ii] - entityVec[lasta1 + ii] - attributeVec[lastar + ii]);
        if (x > 0)
            x = -transeAlpha; 
        else
            x = transeAlpha;
        attributeVec[lastar + ii] -= x;
        entityVec[lasta1 + ii] -= x;
        valueVec[lasta2 + ii] += x;

        x = (valueVec[lastb2 + ii] - entityVec[lastb1 + ii] - attributeVec[lastbr + ii]);
        if (x > 0)
            x = transeAlpha;
        else
            x = -transeAlpha;
        attributeVec[lastbr + ii] -=  x;
        entityVec[lastb1 + ii] -= x;
        valueVec[lastb2 + ii] += x;
    }
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) 
{
    float sum1 = calc_sum(e1_a, e2_a, rel_a); 
    float sum2 = calc_sum(e1_b, e2_b, rel_b); 
    if(sum1 + margin > sum2){
        res += margin + sum1 - sum2;
        gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
    }
}

int a_corrupt_head(int id, int h, int r) 
{
    int lef, rig, mid, ll, rr;

    lef = aLefHead[h] - 1; 
    rig = aRigHead[h];

    while (lef + 1 < rig) 
    {
        mid = (lef + rig) >> 1;
        if (aTrainHead[mid].r >= r) 
            rig = mid; 
        else
            lef = mid;
    }
    ll = rig;

    lef = aLefHead[h];
    rig = aRigHead[h] + 1;
    while (lef + 1 < rig) 
    {
        mid = (lef + rig) >> 1;
        if (aTrainHead[mid].r <= r) 
            lef = mid; 
        else
            rig = mid;
    }
    rr = lef;

    int tmp = rand_max(id, valueTotal - (rr - ll + 1)); 
    if (tmp < aTrainHead[ll].t) 
        return tmp;
    if (tmp > aTrainHead[rr].t - rr + ll - 1) 
        return tmp + rr - ll + 1;

    lef = ll, rig = rr + 1;
    while (lef + 1 < rig) 
    {
        mid = (lef + rig) >> 1;
        if (aTrainHead[mid].t - mid + ll - 1 < tmp)
            lef = mid;
        else
            rig = mid;
    }
    return tmp + lef - ll + 1;
}

int a_corrupt_tail(int id, int t, int r) 
{
    int lef, rig, mid, ll, rr;
    lef = aLefTail[t] - 1;
    rig = aRigTail[t];
    while (lef + 1 < rig) 
    {
        mid = (lef + rig) >> 1;
        if (aTrainTail[mid].r >= r) 
            rig = mid; 
        else
            lef = mid;
    }
    ll = rig;
    lef = aLefTail[t];
    rig = aRigTail[t] + 1;
    while (lef + 1 < rig) 
    {
        mid = (lef + rig) >> 1;
        if (aTrainTail[mid].r <= r) 
            lef = mid; 
        else
            rig = mid;
    }
    rr = lef;
    int tmp = rand_max(id, entityTotal - (rr - ll + 1));
    if (tmp < aTrainTail[ll].h) 
        return tmp;
    if (tmp > aTrainTail[rr].h - rr + ll - 1) 
        return tmp + rr - ll + 1;
    lef = ll, rig = rr + 1;
    while (lef + 1 < rig) 
    {
        mid = (lef + rig) >> 1;
        if (aTrainTail[mid].h - mid + ll - 1 < tmp)
            lef = mid;
        else
            rig = mid;
    }
    return tmp + lef - ll + 1;
}

void* transetrainMode(void *con) 
{ 
	int id;
	id = (unsigned long long)(con);
	next_random[id] = rand(); 
    for (int k = a_transeBatch / transeThreads; k >= 0; k--) 
    {
        int j;
        int i = rand_max(id, a_transeLen);
        int pr = 500;
        int h1, t1, h2, t2,r;
        int EorV_flag=0;
        if (randd(id) % 1000 < pr) 
        { 
            j = a_corrupt_head(id, aTrainList[i].h, aTrainList[i].r);
            train_kb(aTrainList[i].h, aTrainList[i].t, aTrainList[i].r, aTrainList[i].h, j, aTrainList[i].r);
            h1 = aTrainList[i].h, t1 = aTrainList[i].t, r = aTrainList[i].r;
            h2 = aTrainList[i].h, t2 = j;
            norm(valueVec + dimension * j);
            EorV_flag=0;
        } 
        else 
        {
            j = a_corrupt_tail(id, aTrainList[i].t, aTrainList[i].r);
            train_kb(aTrainList[i].h, aTrainList[i].t, aTrainList[i].r, j, aTrainList[i].t, aTrainList[i].r);
            h1 = aTrainList[i].h, t1 = aTrainList[i].t, r = aTrainList[i].r;
            h2 = j, t2 = aTrainList[i].t;
            norm(entityVec + dimension * j);
            EorV_flag=1;
        }
        norm(attributeVec + dimension * aTrainList[i].r);
        norm(entityVec + dimension * aTrainList[i].h);
        norm(valueVec + dimension * aTrainList[i].t);

        if(float(randd(id)%1000)/1000.0 < combinationProbability[h1]) // 硬对齐 a=b , c -x
        { //loss = y + positive - negative
            int h1_cor = correspondingEntity[h1];//真实数据 a-b 0.5 可能模型认为 a-c 0.3 a-d 0.6
            //h1=h1_cor;
            train_kb(h1_cor, t1, r, h2, t2, r);
            norm(entityVec + dimension * h1_cor);
        }
        if(float(randd(id)%1000)/1000.0 < combinationProbability[h2])
        {
            int h2_cor = correspondingEntity[h2];
            train_kb(h1, t1, r, h2_cor, t2, r);
            norm(entityVec + dimension * h2_cor);
        }
        norm(attributeVec + dimension * aTrainList[i].r);
        norm(entityVec + dimension * aTrainList[i].h);
        norm(valueVec + dimension * aTrainList[i].t);
        if(EorV_flag==1)
            norm(entityVec + dimension * j);
        else
            norm(valueVec+dimension*j);
    }
}

double calc_distance(int ent1, int ent2)
{ 
	double sum=0;
	if (L1_flag)
		for (int ii=0; ii<dimension; ii++)
		    sum+=fabs(entityVec[ent1*dimension + ii]-entityVec[ent2*dimension + ii]); 
	else
		for (int ii=0; ii<dimension; ii++)
		    sum+=pow(entityVec[ent1*dimension + ii]-entityVec[ent2*dimension + ii], 2); 
	return sum;
}

void do_combine()
{
    time_t beginTimer, endTimer; 
    time(&beginTimer); 
    printf("Combination begins.\n");

    vector<pair<double, pair<int, int> > > distance2entitiesPair; 
    for(auto &i : entitiesInKg1) // kg1 <-> kg2
    	for(auto &j : entitiesInKg2)
    	    distance2entitiesPair.push_back(make_pair(calc_distance(i, j), make_pair(i, j))); 
    sort(distance2entitiesPair.begin(), distance2entitiesPair.end()); 
    
    set<int> occupied;
    float minimalDistance = 0;

    for(auto &i : distance2entitiesPair)
    { 
    	if(i.first > 0)
    	{ 
    		minimalDistance = i.first;
    		break;
    	}
    }
//1. init
//2.
    correspondingEntity.clear(); 
    fill(combinationProbability.begin(), combinationProbability.end(), 0); 
    int combination_counter = 0;
    for(auto &i: distance2entitiesPair)
    {
    	int dis = i.first;
    	int ent1 = i.second.first;
    	int ent2 = i.second.second;
    	if(dis > combination_threshold)
    		break;
        if(occupied.count(ent1) || occupied.count(ent2))
            continue; 
        correspondingEntity[ent1] = ent2;
        correspondingEntity[ent2] = ent1;
        occupied.insert(ent1);
        occupied.insert(ent2);
        combinationProbability[ent1] = sigmoid(combination_threshold - dis); 
        combinationProbability[ent2] = sigmoid(combination_threshold - dis);
        if(combination_counter == combination_restriction) 
        	break;
        combination_counter++;
    }
    time(&endTimer); 
    printf("Using %.f seconds to combine %d entities pairs.\n", difftime(endTimer, beginTimer), combination_counter);
    combination_restriction += 1000; 
}

void* train_transe(void *con) 
{
    a_transeLen = aTripleTotal;
    a_transeBatch = a_transeLen /nbatches;
	next_random = (unsigned long long *)calloc(transeThreads, sizeof(unsigned long long)); 

    for (int epoch = 0; epoch < transeTrainTimes; epoch++) 
    { 

    	if(epoch > 999 && epoch % 500 == 0)
    	{ 
    	    do_combine();
    	}
    	res = 0;
    	for (int batch = 0; batch < nbatches; batch++) 
    	{
    	    pthread_t *pt = (pthread_t *)malloc(transeThreads * sizeof(pthread_t)); 
    	    for (int a = 0; a < transeThreads; a++)
    	        pthread_create(&pt[a], NULL, transetrainMode,  (void*)a);
    	    free(pt);
    	}
    	printf("epoch %d %f\n", epoch, res);
    	fflush(stdout);
    }
}

void out_transe(string iter = "") 
{
    FILE* f3 = fopen((outPath + "entity2vec" + iter + ".bern").c_str(), "w");
    for (int  i = 0; i < entityTotal; i++) 
    {
    	int last = i * dimension;
    	for (int ii = 0; ii < dimension; ii++)
    		fprintf(f3, "%.6f\t", entityVec[last + ii] );
    	fprintf(f3,"\n");
    }
    fclose(f3);
}

int main() 
{
    cout<<"00"<<endl;

    srand(19950306);
    init();
    cout<<"11"<<endl;
    train_transe(NULL);
    out_transe();
    return 0;
}