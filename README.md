# H.D.A.I.2021 Track 2. 심전도 데이터셋 README
## 팀명: KNU_챌린저
gitnub 주소: https://github.com/suhyeon0123/HDAI

# 환경
## 사용 모듈
- torch : 1.9.0+cu111
- numpy : 1.20.3
- sklearn : 0.23.2
- xmltodict (pip install xmltodict를 통해 설치해주세요.)
# 모델 설명
저희는 CNN-1D를 기반으로 한 모델을 사용했습니다. CNN 기반의 모델을 사용한 이유는 데이터 전처리에 있습니다.


첫 번째로는 데이터의 특성입니다. 부정맥 데이터는 전체적인 맥락에서 잘못된 이미지 하나만을 확인하면 됩니다. 다시 말하자면, 심전도 데이터셋 경우에는 국소적으로 다른 부분만 탐색하면 됩니다. 이러한 이유로 전반적인 이미지를 확인하는 CNN이 우수하지 않았나 예상합니다.

데이터 전처리에서, 데이터마다 lead의 개수가 다르다는 것을 확인했습니다. 저희는 8개인 read를 심전도의 특정 공식을 이용하여 12개를 채워서 모든 데이터가 12개의 lead를 갖도록 전처리를 하였습니다.

처음엔 RNN을 사용하여 부정맥을 확인하려고 했습니다. 그러나 AUC가 90%밖에 나오지 않았습니다. 두 번째 모델로는 CNN을 사용했습니다. 결과는 AUC 98%정도의 성능을 보여줬습니다.
저희는 RNN모델보다 CNN모델이 더 좋은 성능을 보여준 이유로 두 가지를 제시합니다. 

첫 번째로는 데이터의 특성입니다. 위에서 설명드린 것 처럼 부정맥 데이터는 전체적인 맥락에서 잘못된 이미지 하나만을 확인하면 됩니다. 다르게 말하자면, 일반적인 시계열 데이터의 경우 데이터의 앞뒤의 맥락을 확인하여 현재 부분이 전반적인 흐름과 다른 지 확인합니다. 그러나 심전도 데이터셋 경우에는 국소적으로 다른 부분만 탐색하면 됩니다. 이러한 이유로 인해서 앞뒤의 맥락을 확인하는 RNN보다는 CNN이 우수한 성능을 보인 것으로 확인됩니다.

두 번째로 RNN의 경우 sequence가 길어질수록 앞의 내용을 잊어버리는 경향이 있습니다. 이번에 주어진 심전도 데이터의 경우, sequence의 길이가 5000으로, 일반적으로 RNN에서 사용하는 sequence보다 50배나 길었습니다. 위와 같은 이유로 RNN의 성능이 더 낮아졌음을 예상합니다.
이러한 이유가 저희가 기존에 이상치 탐색에서 많이 쓰던 RNN이 아닌 CNN을 선택한 이유입니다.

# 실행 가이드
train
```python
python main.py --path [dataset path] --gpu [the number of gpu to use] --lr [learning rate] --dropout [dropout] --batch_size [batch size] --training_epoch [training epoch]
```

test 
```python
python main.py --path [dataset path] --gpu [the number of gpu to use] --lr [learning rate] --dropout [dropout] --batch_size [batch size] --test --test_model_weights [model_path]
```

<!-- ```python -->
## parser
**--path**: dataset path, type=str</br>
**--gpu**: the number of gpu to use, type=int, default=0</br>
**--lr**: learning rate, type=float, default=5e-6</br>
-**-dropout**: drop out, type=float, default=0.3</br>
**--batch_size**: batch size, type=int, default=32</br>
**--training_epoch**: training epoch, type=int, default=100</br>

# 데이터 분석
## ECG

확장자: .xml

데이터 구조:

┏ RestingECG</br>
┃　┏ PatientDemographics - 환자 정보 [PatientID]</br>
┃　┃　PatientID - 환자 ID</br>
┃　┗━</br>
┃　┏ Waveform - 파형 정보</br>
┃　┃　WaveformType - 파형 종류 ('Median', 'Rhythm')</br>
┃　┃　WaveformStartTime - 파형 시작 시간 [0]</br>
┃　┃　NumberofLeads - LeadData의 개수 (8, 11, 12)</br>
┃　┃　SampleType - ['CONTINUOS_SAMPLES']</br>
┃　┃　SampleBase - [500]</br>
┃　┃　SampleExponent - [0]</br>
┃　┃　HighPassFilter - (1, 16, 56, X)</br>
┃　┃　LowPassFilter - (40, 150, X)</br>
┃　┃　ACFilter - ('NONE', 60, X)</br>
┃　┃　┏ LeadData - 리드 데이터 : NumberofLeads만큼의 개수로 이뤄진 리스트 형태로 돼있다.</br>
┃　┃　┃　LeadByteCountTotal - (1200, 10000)</br>
┃　┃　┃　LeadTimeOffset - 오프셋 [0]</br>
┃　┃　┃　LeadSampleCountTotal - WaveFormData의 데이터 개수(샘플 길이) (600, 5000)</br>
┃　┃　┃　LeadAmplitudeUnitsPerBit - (-10, 4.88, 5)</br>
┃　┃　┃　LeadAmplitudeUnits - 진폭 단위 ['MICROVOLTS']</br>
┃　┃　┃　LeadHighLimit - 최댓값 (32767, 2147483647)</br>
┃　┃　┃　LeadLowLimit - 최솟값 (-32768, 268435456)</br>
┃　┃　┃　LeadID - 리드 할당 ID ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF')</br>
┃　┃　┃　LeadOffsetFirstSample - [0]</br>
┃　┃　┃　FirstSampleBaseline - [0]</br>
┃　┃　┃　LeadSampleSize - [2]</br>
┃　┃　┃　LeadOff - ('FALSE', 'TRUE')</br>
┃　┃　┃　BaselineSway - ('FALSE', 'TRUE')</br>
┃　┃　┃┃ExcessiveACNoise - ('FALSE', 'TRUE')</br>
┃　┃　┃┃MuscleNoise - ('FALSE', 'TRUE')</br>
┃　┃　┃　LeadDataCRC32</br>
┃　┃　┃　WaveFormData - 파형 데이터. b64로 디코드하여 수치화할 수 있다.</br>
┃　┃　┗━</br>
┃　┗━</br>
┗━</br>

### 구조 종류
데이터 구조를 바탕으로 임의의 이름을 부여하였다.

1. 2-Wave Normal</br>
    구조:</br>
    Median과 Rhythm 두 Waveform을 갖고 있다. 각 Waveform은 8개의 리드를 사용한다.

    주의점:</br>
    Waveform이 Median, Rhythm 두 종류가 리스트 형태로 되어 있기 때문에 따로 Rhythm만 추출할 필요가 있다. (해결)</br>

    사용 파일:</br>
    `train/arrhythmia/5_2_*`</br>
    `train/normal/5_0_*`</br>
    `validation/arrhythmia/8_2_007743_ecg.xml`</br>
    `validation/arrhythmia/8_2_007824_ecg.xml`</br>
    `validation/arrhythmia/8_2_008124_ecg.xml`</br>

2. Simple</br>
    구조:</br>
    Waveform이 NumberofLeads, SampleBase, LeadData만 사용하는 단순한 구조를 갖는다.</br>
    (Filter 세 개도 레이블은 있으나 값이 존재하지 않는다.)</br>
    총 12개의 리드를 사용하고, 각 리드는 샘플 길이와 UnitsPerBit, Units, ID, SampleSize 데이터로만 이루어져 있다.</br>

    사용 파일:</br>
    `train/arrhythmia/6_2_*`</br>

3. Simple middle NA</br>
    구조:</br>
    Simple 구조와 같은 구조를 가지나, 6, 7, 8번 리드의 데이터가 누락돼있다.</br>

    주의점:</br>
    누락된 리드의 데이터를 어떻게 처리할 것인가? (해결)</br>

    사용 파일:</br>
    `train/arrhythmia/6_2_001267_ecg.xml`</br>
    `train/arrhythmia/6_2_003198_ecg.xml`</br>
    `train/arrhythmia/6_2_004827_ecg.xml`</br>
    `train/arrhythmia/6_2_004971_ecg.xml`</br>

4. Simple last NA</br>
    구조:</br>
    Simple 구조와 같은 구조를 가지나, 9, 10, 11번 리드의 데이터가 누락돼있다.</br>

    주의점:</br>
    위와 동일</br>

    사용 파일:</br>
    `train/arrhythmia/6_2_001970_ecg.xml`</br>
    `train/arrhythmia/6_2_003131_ecg.xml`</br>
    `train/arrhythmia/6_2_004979_ecg.xml`</br>

5. 11 Simple</br>
    구조:</br>
    Simple 구조와 같은 구조를 가지나 리드의 개수가 11개이다.</br>

    주의점:</br>
    위와 동일</br>

    사용파일:</br>
    `train/arrhythmia/6_2_004491_ecg.xml`</br>

6. Ex Normal</br>
    구조:</br>
    8개의 리드를 사용하는 Normal과 같은 구조를 가지나 각 리드에 "ExcessiveACNoise" 와 "MusicleNoise"라는 새로운 값이 추가돼있다.</br>

    사용파일:</br>
    `train/arrhythmia/8_2_*`</br>
    `validation/arrhythmia/8_2_*`</br>

7. Normal</br>
    구조:</br>
    특이사항이 없다. 8개의 리드를 사용.</br>

    사용파일:</br>
    `train/arrhythmia/8_2_000220_ecg.xml`</br>
    `train/arrhythmia/8_2_004888_ecg.xml`</br>
    `validation/arrhythmia/8_2_007743_ecg.xml`</br>
    `validation/arrhythmia/8_2_007824_ecg.xml`</br>
    `validation/arrhythmia/8_2_008124_ecg.xml`</br>

## Label
0. Normal sinus rhythm
   
1. Sinus tachycardia
2. Atrial fibrillation
3. Atrial flutter
4. Premature atrial complex
5. Ectopic atrial rhythm
6. Supraventricular tachycardia
7. Premature ventricular complex
8. Idioventricular rhythm
9.  Ventricular tachycardia
10. 1st degree AVB
11. 2nd degree AVB (Mobitz type 1)
12. 2nd degree AVB (Mobitz type 2)
13. 3rd degree (complete AV block)
14. Sinus bradycardia
15. Junctional rhythm
