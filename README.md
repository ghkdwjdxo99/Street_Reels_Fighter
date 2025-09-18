# Street_Reels_Fighter

## 프로젝트 목표
릴스에서 유행하는 춤을 Pose Estimation을 통해 연습 및 대결할 수 있게 하는 기능을 제공하는 게임 시스템입니다.

### 주요 기능
- YOLOv8-Pose Model을 사용해 사용자와 비교 영상에서 Key Point 추출
- 추출한 Key Point로 관절 각도를 계산하여 유사도 측정
- Key Point와 각 아바타 파츠들의 관절 좌표 값을 매칭하여 아바타 영상 생성
- STM32와 UART 통신으로 서보모터를 조작해 사용자를 추적하는 Pan 구현


## 구성도
<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/10bc5493-2cbd-427e-8ccc-1254b1499612" />




## 흐름도
<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/88f75fc4-9132-4528-b9f9-8a906baca271" />



## 팀 역할
<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/43a84000-a948-4d23-b2fc-cde58099058e" />



### Key Point 추출
<img width="878" height="936" alt="image" src="https://github.com/user-attachments/assets/ea55223d-63bd-49b0-9c76-c5e92dbff430" />


### 관절 각도 계산
<img width="1431" height="926" alt="image" src="https://github.com/user-attachments/assets/f5e19040-a71d-4c29-85e4-006b205297dc" />


### 아바타 생성
<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/8aa3ff1c-1142-47d3-ad26-6116e9ac6e31" />


### 카메라 Pan (STM32)
<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/b994d77f-d89b-4535-a556-3a9f32b39cfb" />
<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/5e236b3a-aa04-439d-9fdc-c2151ecf3bc1" />




## 시연 영상


https://github.com/user-attachments/assets/3b85f136-1a0b-4b7d-9cec-c0dcfc4e700a





## 전체 자료
[스릴파_뚝's딱's_상세 발표 자료.pdf](https://github.com/user-attachments/files/22405165/_.s.s_.pdf)
