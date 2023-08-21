# crowd-density-detection-ai-project
사람탐지 모델을 기반으로(어떤 모델을 쓸지는 미정), 측정하는 사진에 대해 군중밀집사고의 위험도가 얼마나 되는지 측정하는 모델입니다.

## 프로젝트 참여자
* 박세진, 강재훈, 이재현, 정준, 정준현

## 라이센스
* 코드는 저희가 참조한 코드의 라이센스가 파생코드에 대해 특정 라이센스를 강제하지 않는 한, 프로젝트 참여자 외에 MIT 라이센스를 적응하는 것으로 간주합니다.
* 이미지는 웹에서 무작위로 취득한 것이므로, 라이센스 여부를 주장하지 않습니다.
* 이 레포지토리에서 사용한 자료의 일부가 어떤 저작권에 대해 침해하는 일이 발생했다면 `Issues` 항목에 관련 내용을 안내해주시면 협조하겠습니다.

# 모델 형태
## 폴더 구조
* dataset_raw: 이미지 분류하기 전 기본 이미지들입니다. 이 이미지들은 라벨링하기 전, 형식에 맞게 가공해야합니다.
* dataset_refined: 라벨링이 완료되지 않은 이미지들입니다. 이미지 자체는 기본적으로 `dataset_labeled`와 동일하지만, 라벨이 되어있지 않기 때문에 어떤 정보도 담겨있지 않습니다. 이 파일은 라벨링 소프트웨어에 활용할 수 있습니다. 해당 데이터셋에 라벨링이 완료가 되면, 형식에 맞게 다시 가공하여 `dataset_labeled`로 복사합니다.
* dataset_labeled: 이미지와 라벨이 매치되어 작업이 완료된 데이터셋입니다.
  * train_images: 학습용 이미지
  * train_labels: 학습용 이미지에 대한 라벨
  * valid_images: 검증용 이미지
  * valid_labels: 검증용 이미지에 대한 라벨

아직 확실하게 정리된 구조가 아니니 참고만 바랍니다.
