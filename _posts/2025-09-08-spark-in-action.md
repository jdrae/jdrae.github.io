---
title: "스파크를 다루는 기술: Spark in Action"
date: 2025-09-08
categories: [Data engineering, Spark]
tags: [data engineering, spark, book review]
---

## 스파크란?

하둡의 맵리듀스를 대체하는 빅데이터 처리 플랫폼.
하둡은 분산 컴퓨팅용 자바 기반 오픈소스 프레임워크. (= 하둡 분산 파일 시스템 hdfs + 맵리듀스 처리엔진)
스파크는 범용 분산 컴퓨팅 플랫폼이라는 점에서 하둡과 유사하지만, 대량의 데이터를 메모리에 유지하는 설계로 계산 성능 향상.

### 단점

분산 아키텍처 때문에 처리 시간에 약간의 오버헤드가 발생. 대량의 데이터셋에서는 무시할 수 있지만, 작은 데이터셋은 다른 프레임워크가 효율적.
OLTP(online transaction processing), 즉 대량의 원자성 트랜잭션에는 적합하지 않다. (대신 일괄 처리 작업인 olap 는 적합)

### 하둡

- 병렬처리 parallelization: 여러 연산을 잘게 나눔
- 데이터 분산 distribution: 여러 노드로 분산
- 장애 내성 fault tolerance: 분산 컴포넌트의 장애에 대응

※ 계산 결과를 다른 잡에서 사용하려면 hdfs 에 저장후 가져와야해서 반복 알고리즘에는 부적합

※ 모든 문제를 맵리듀스 연산으로 분해할 수 없음


## 실행과정과 기본 개념

300MB 파일이 HDFS 클러스터에 저장하면, 클러스터의 노드 3개에 128, 128, 44MB 의 블록으로 각각 저장한다. (복제계수를 기본값 3으로 설정하면 HDFS 는 노드에 저장한
각 블록을 다른 노드 두 개에 복제한다.)

스파크는 파일의 각 블록(=파티션)이 저장된 위치를 하둡에게 요청한 후, 각 블록을 해당 블록이 저장된 hdfs 노드의 ram 메모리에 로드한다. → 데이터 지역성 data locality ⇨ 대량의 데이터를 네트워크로 전송하지 않아도 됨

파티션의 집합이 RDD 가 참조하는 분산 컬렉션이고, 사용자는 컬렉션이 여러 노드에 분산 저장된다는 사실을 알 필요 없다.

만약 필터링을 한다면 필터링된 정보만 RAM 에 저장되고, cache 를 통해 파일을 다시 로드할 필요없이 다른 잡에서도 RDD 가 메모리에 유지되도록 지정할 수 있다. 이때 필터링은 노드 세개에서 병렬로 실행된다.

### RDD

- 불변성 immutable: 읽기 전용 read-only
    - 변환 연산자는 항상 새로운 RDD 객체를 생성. 즉, 한번 생성된 rdd 는 불변
- 복원성 resilient: 장애 내성
    - 노드에 장애가 발생해도 RDD 원복 가능
    - 데이터셋을 만드는데 사용된 변환 연산자의 로그를 남겨 장애 발생 시 해당 노드가 가진 데이터셋만 다시 계산해 RDD 복원
- 분산 distributed: 노드 한 개 이상에 저장된 데이터셋
    - 투명성 location transperency: 파일의 물리적인 섹터를 여러 장소에 저장해도 파일 이름만으로 파일에 접근 가능.
    - 물리적인 상황을 논리적 개념으로 추상화
- 변환 연산자: 데이터를 조작해 새로운 RDD 생성 (ex. filter, map)
    - 지연 실행 lazy evaluation: 행동 연산자를 호출하기 전까지는 계산을 실행하지 않음.
- 행동 연산자: 계산 결과를 반환 (ex. count, foreach)

### for comprehensions

```scala
val employess = Set() ++ (
    for {
        line <- fromFile(empPath).getLines
    } yield line.trim // for 루프의 각 사이클별로 값 하나(line.trim)를 임시 컬렉션에 추가한다. 임시 컬렉션은 루프가 종료될때 반환 후 삭제.
)
```

### 공유 변수

```scala
val bcEmployees = sc.broadcast(employees)
val isEmp = user => bcEmployees.value.contains(user)
```

- 공유 변수는 클러스터의 각 노드에 정확히 한 번만 전송하고, 메모리에 자동으로 캐시된다. (그렇지 않으면 작업을 수행할 태스크 개수만큼 반복적으로 네트워크에 전송하게 된다)
- P2P 프로토콜로 공유 변수 전파하므로, 마스터 실행이 크게 지연되지 않는다. (서로 공유 변수 교환하며 확산시킴 = 가심 프로토콜 gossip protocol)
- 반드시 value 메서드로 접근한다.


## 데이터 파티셔닝

데이터를 여러 클러스터 노드로 분할 → 성능과 리소스 점유량 좌우함
RDD 의 파티션은 RDD 데이터의 일부. 스파크는 파일을 파티션으로 분할해 클러스터 노드에 분산 저장한다. 분산된 파티션이 모여 RDD 를 형성한다.

파티션 개수:
클러스터에 분배하는 과정에서 영향을 미침. RDD 에 변환 연산을 실행할 태스크 개수와도 직결.
파티션이 적으면 클러스터를 충분히 활용할 수 없다. 또 각 태스크가 처리할 데이터 분량이 실행자의 메모리 리소스를 초과하기도 함.
⇨ 클러스터의 코어 개수보다 3~4배 많은 파티션을 사용하는게 좋음. ※ 태스크가 너무 많으면 태스크 관리 작업에 병목 발생

### Partitioner

RDD 의 각 요소에 파티션 번호를 할당하면서 파티셔닝 수행.

**HashPartitioner**
기본값.
각 요소의 자바 해시코드를 mod 공식으로 계산(partitionIndex = hashCode % numOfPartitions)
무작위이기 때문에 모든 파티션을 정확하게 같은 크기로 분할할 가능성이 낮다. (상대적으로 적은 수의 파티션으로 나누면 대체로 고르게 됨

**RangePartitioner**
정렬된 RDD 의 데이터를 거의 같은 범위 간격으로 분할. 샘플링한 데이터를 기반으로 범위 경계를 결정. (거의 사용 x)

**Pair RDD 의 사용자 정의 Partitioner**
키-값 쌍 Pair RDD 데이터를 처리할때 특정 기준에 따라 정확하게 배치할 수 있다.

### 셔플링

셔플링은 파티션 간의 물리적인 데이터 이동을 의미한다.
새로운 RDD 의 파티션을 만들려고 여러 파티션의 데이터를 합칠때 발생.

```scala
val prods = transByCust.aggregateByKey(List[String]())(
    (prods, tran) => prods ::: List(tran(3)),
    (prods1, prods2) => prods1 ::: prods2
)
```


ex. 키를 기준으로 그루핑하려면 RDD 의 파티션을 모두 살펴보고 키가 같은 요소를 전부 찾은 후 물리적으로 묶어서 새로운 파티션 구성

1. 변환함수: 파티션 별로 값을 병합해 값의 타입을 변경함
2. 병합함수: 셔플링 단계를 거치며 여러 값을 하나로 최종 병합
⇨ 셔플링 바로 전에 수행한 태스크가 맵 태스크, 바로 다음에 수행한 태스크가 리듀스 태스크.

![](/assets/posts/2025-09-08-spark-in-action/1.jpg){:width="500px"}

**외부 셔플링 서비스**
셔플링을 수행하면 실행자는 다른 실행자의 파일을 pulling 으로 읽어들여야한다. 하지만 도중 장애가 발생하면 해당 실행자가 처리한 데이터를 가져올 수 없어 중단된다.
외부 셔플링 서비스는 실행자가 중간 셔플 파일을 읽을 수 있는 단일 지점을 제공해 데이터 교환 과정을 최적화 한다.

**매개변수** 예시

- spark.shuffle.manager: 셔플링 알고리즘. hash, sort(기본)
- spark.shuffle.consolidateFiles: 셔플링 중 생성된 중간 파일의 통합 여부를 지정한다. 기본 false.
- spark.shuffle.spill: 메모리 리소스의 제한 여부. 초과 시 데이터를 디스크로 내보낸다. 기본 true.

### 불필요한 셔플링 줄이기

**셔플링 발생 조건**

- Partitioner 를 명시적으로 변경하는 경우
    - 사용자 정의 Partitioner 를 쓰거나, 이전 HashPartitioner 와 파티션 개수가 다른 HashPartitioner 를 사용할 경우 셔플링 발생
    ⇨ 가급적 기본 Partitioner 를 사용한다
- Partitioner 를 제거하는 경우
    - map, flatMap 은 Partitioner 를 제거한다. 이후 join, groupByKey 같은 연산자를 사용하면 셔플링이 발생한다.
    ⇨ 키를 변경할 필요가 없다면 mapValues, flatMapValues 를 사용한다.
    ⇨ 파티션 내에서만 데이터가 매핑되도록 mapPartitions, mapPartitionsWithIndex, glom 등 사용해 preservePartitioning=true 로 설정

### RDD 파티션 변경

작업 부하를 분산시키기 위해 파티셔닝을 명시적으로 변경해야할 때가 있다.

- partitionBy
    - PairRDD 에서만 사용하고 파티셔닝에 사용할 Partitioner 객체를 인자로 전달하여 새로운 RDD 생성
- coalesce
    - 파티션 개수를 변경하기 위해 사용.
    - 줄일때: 새로운 파티션 개수와 동일한 개수의 부모 RDD 파티션을 선정하고, 나머지 파티션의 요소를 분할해 병합.
    - shuffle false: coalesce 이전의 변환 연산자도 현재 파티션 개수로 사용
    - shuffle true: coalesce 이전의 변환 연산자는 원래의 파티션 개수를 사용하고, 그 이후만 현재 파티션 개수를 사용
- repartition
    - shuffle 을 true 로 설정해 coalesce 를 호출한 결과
- repartitionAndSortWithinPartition
    - 새로운 Partitioner 를 받아 각 파티션 내에서 요소를 정렬한다. 셔플링 단계에서 정렬작업을 함께 수행.
    - repartition 호출한 후 정렬하는 것보다 성능이 좋음

### RDD 의존 관계

스파크의 실행 모델은 DAG = RDD 를 정점으로 RDD 의존관계를 간선으로 정의한 그래프. 변환 연산자 호출할때마다 새로운 간선 생성.
새 RDD 가 이전 RDD 에 의존하고, 이 그래프를 RDD lineage 라고 한다.

- 좁은 의존관계: 데이터를 다른 파티션으로 전송할 필요가 없는 변환 연산
    - 1:1 의존관계: union 외 모든 것
    - 범위형 의존관계: 여러 부모 RDD 에 대한 의존 관계를 하나로 결합 = union 만 해당
- 넓은 의존관계(shuffle): 셔플링 수행 시 형성. join 시 반드시 형성됨

**스테이지**
스파크는 셔플링이 발생하는 지점을 기준으로 스파크 잡 하나를 여러 스테이지로 나눈다.
스테이지 결과는 중간 파일의 형태로 실행자 머신의 디스크에 저장된다.
각 스테이지와 파티션별로 태스크를 생성해 실행자에 전달한다.
스테이지가 셔플링으로 끝나는 경우 셔플-맵 태스크라고 한다.
마지막 스테이지에 생성된 태스크를 결과 태스크라고 한다.

**체크포인트**
계보가 너무 길 경우 복구하기 힘들기 때문에 RDD 데이터 전체를 중간에 저장함. 장애 시 해당 시점부터 복구.


## 스파크 런타임 컴포넌트

![](/assets/posts/2025-09-08-spark-in-action/2.heic){:width="300px"}

![](/assets/posts/2025-09-08-spark-in-action/3.heic){:width="300px"}

**Client**

드라이버를 시작하는 spark-submit, spark-shell, 스파크 API 를 사용한 커스텀 애플리케이션.

**Driver**

스파크 애플리케이션에 하나만 존재하는 일종의 래퍼 역할.

- 클러스터 매니저에 리소스 요청 (메모리, CPU)
- 애플리케이션 로직을 스테이지와 태스크로 분할
- 여러 executor 에 태스크 전달
- 태스크 실행 결과 수집

드라이버가 어디서 실행되냐에 따라 다음과 같은 두 모드가 있다

1. 그림1) 클러스터 배포 모드
    
    드라이버가 클라이언트와 분리된다.
    
    클러스터 내부에서 별도의 JVM 프로세스로 실행되기에, 드라이버 프로세스의 리소스(JVM 힙 메모리)를 클러스터가 관리한다.
    
2. 그림2) 클라이언트 배포 모드
    
    드라이버를 클라이언트의 JVM 프로세스에서 실행한다.
    

**Executor**

드라이버가 요청한 태스크를 실행하고 결과를 다시 드라이버로 반환하는 JVM 프로세스.

태스크들을 여러 태스크 슬롯에서 병렬로 실행한다.

일반적으로 태스크 슬롯은 스레드로 구현되므로 CPU 코어의 2~3배를 개수로 설정한다.

**스파크 컨텍스트**

스파크의 런타임 인스턴스에 접근할 수 있는 기본 인터페이스로, 여러 메소드를 제공한다.

드라이버는 SparkContext 인스턴스를 생성하고 시작한다.

스파크 API 로 애플리케이션을 실행할때는 애플리케이션에서 직접 스파크 컨텍스트를 실행해야한다. 

JVM 당 하나만 생성할 수 있다. (여러개 쓰는 옵션이 있지만 테스트용이라 권장하지 않음)



## 스케줄링

1. executor(JVM 프로세스)와 CPU(태스크 슬롯) 리소스 스케줄링
2. 클러스터 매니저가 CPU, 메모리 리소스를 각 executor 에 할당
3. 애플리케이션 내부에서 잡 스케줄링 실행

### 클러스터 리소스 스케줄링

단일 클러스터에서 실행하는 여러 스파크 애플리케이션의 실행자에 리소스 할당.

클러스터 매니저: 프로세스 시작/중지/재시작. executor 의 최대 CPU 코어 개수 제한

- 드라이버가 요청한 executor 프로세스 시작
- (클러스터 배포 모드일 경우) 드라이버 프로세스를 시작

※ 애플리케이션 간 executor 는 공유되지 않음. 여러 애플리케이션을 한 클러스터에서 동시에 실행하면 리소스 경쟁 발생.

### 스파크 잡 스케줄링

단일 스파크 애플리케이션 내에서 태스크를 실행할 CPU, 메모리 리소스를 스케줄링

드라이버에는 다수의 스케줄러 객체가 있음. executor 가 실행되면 어떤 executor 가 어떤 태스크 수행할지 결정.

동일한 SparkContext 를 공유하는 여러 잡이 executor 의 리소스를 놓고 경쟁. (SparkContext 는 thread-safe).

클러스터의 CPU 리소스 사용량을 좌우한다.

메모리 사용량에도 간접적 영향(더 많은 태스크를 단일 JVM 에서 실행할 수록 더 많은 힙 메모리 사용)

※ CPU 리소스는 태스크 단위로 관리. 메모리 리소스는 여러 세그먼트로 분리해서 관리.

**FIFO 선입선출 스케줄러**

가장 먼저 리소스를 요청한 잡이 필요한 만큼 태스크 슬롯을 차지.

잡이 리소스를 많이 차지하지 않다면 동시에 실행될 수 있지만, 모두 차지해야한다면 기존 잡이 리소스를 다 사용할때까지 다음 잡은 대기해야한다.

**FAIR 공정 스케줄러**

round-robin 방시으로 균등하게 리소스 분배.

태스크 슬롯을 더 늦게 요청하더라도 오래 걸리는 잡이 완료될때까지 대기하지 않아도 된다.

스케줄러 풀 기능으로 가중치와 최소 지분을 설정할 수 있다. 가중치를 설정하면 특정 풀의 잡이 다른 풀보다 리소스를 더 많이 할당받을 수 있다. 최소 지분은 각 풀이 항상 사용할 수 있는 최소한의 CPU 코어 개수이다.

**태스크 예비 실행 speculative execution**

리소스가 없어 동일 스테이지의 다른 태스크보다 더 오래 걸리는 낙오 태스크(stranggler task) 문제를 해결.

해당 파티션 데이터를 처리하는 동일한 태스크를 다른 실행자에도 요청해, 기존 태스크가 지연되고 예비 태스크가 완료되면 예비 태스크의 결과를 사용하여 지연 방지.

spark.speculation=True. 

spark.speculation.interval: 예비 태스크 실행 여부 체크. 

spark.speculation.quantile: 예비 태스크 실행 전 완료해야 할 태스크 진척률

spark.speculation.multiplier: 기존 태스크의 지연 정도.

※ 예비 태스크는 잘 판단해야함. 예를 들어 DB 에 데이터 기록할 때 동일 데이터를 중복 기록할 수 있음. 

## 데이터 지역성 data locality

데이터와 최대한 가까운 위치의 실행자에서 태스크를 실행하려고 노력하는 것.

**선호 위치 preferred location**

스파크에는 각 파티션별로 파티션 데이터를 저장한 호스트네임 또는 실행자 목록을 갖고 있다. 이 위치를 참고해 데이터와 가까운 곳에서 연산을 실행할 수 있다. 

※ HDFS 데이터로 생성한 RDD 와 캐시된 RDD 만 선호 위치 정보를 알 수 있다.

HDFS RDD 는 하둡 API 로 HDFS 클러스터의 위치 정보를 가져온다.

캐시된 RDD 는 각 파티션이 캐시된 실행자 위치를 직접 관리한다.

**데이터 지역성 레벨**

최선의 태스크 슬롯을 확보하지 못했을 때, 일정 시간을 기다리다 차선으로 스케줄링 시도.

태스크의 실행 위치에 따라:

- PROCESS_LOCAL: 파티션을 캐시한 실행자
- NODE_LOCAL: 파티션에 직접 접근할 수 있는 노드
    - 네트워크를 통하지 않고 접근할 수 있는 위치. 동일 머신의 다른 실행자
- RACK_LOCAL: 파티션을 저장한 머신과 동일한 랙에 장착된 다른 머신
    - yarn 만 클러스터의 rack 정보를 참고할 수 있어서, yarn 에서만 해당 레벨 가능
    - 랙은 서버 및 네트워크 장비를 장착하는 표준 크기의 프레임. 네트워크로 전송해도 switch 만 거치면 됨
- NO_PREF: 선호하는 위치가 없을 경우 (클러스터 어디서나 동일한 속도로 데이터에 접근)
- ANY: 데이터 지역성을 확보하지 못했을 경우 다른 위치에서 태스크 실행

### 메모리 스케줄링

클러스터 매니저가 executor JVM 프로세스의 메모리 할당 → 스파크가 잡과 태스크가 사용할 메모리 스케줄링

**클러스터 매니저가 관리하는 메모리**

executor 에 할당할 메모리는 spark.executor.memory 로 설정

**스파크가 관리하는 메모리**

스파크 1.5.2 이하:

executor 메모리를 나눠 캐시 데이터와 임시 셔플링 데이터 저장. 나눠진 메모리의 사용량의 초과될 수 있으므로 safety 비율을 정해 default 로 캐시 54%, 셔플링 16%, 나머지 30% 으로 기타 자바 객체와 리소스 저장.

스파크 1.6.0에서 변경:

메모리를 통합해 관리하므로 셔플링 없을 경우 캐시가 전체 메모리를 차지할 수 도 있음. 단, 실행 메모리가 차지한 영역은 스토리지 메모리 영역으로 변경 불가.



## 예제: 실시간 대시보드

![](/assets/posts/2025-09-08-spark-in-action/4.heic){:width="600px"}

```python
class KafkaProducerWrapper(object):
        producer = None
        @staticmethod
        def getProducer(brokerList):
            if KafkaProducerWrapper.producer == None:
                KafkaProducerWrapper.producer = KafkaProducer(bootstrap_servers=brokerList, key_serializer=str.encode, value_serializer=str.encode)
            return KafkaProducerWrapper.producer

if __name__ == "__main__":
    # ... 생략
    
  #data key types for the output map
  SESSION_COUNT = "SESS"
  REQ_PER_SEC = "REQ"
  ERR_PER_SEC = "ERR"
  ADS_PER_SEC = "AD"

  requests = reqsPerSecond.map(lambda sc: (sc[0], {REQ_PER_SEC: sc[1]}))
  errors = errorsPerSecond.map(lambda sc : (sc[0], {ERR_PER_SEC: sc[1]}))
  finalSessionCount = sessionCount.map(lambda c : (long((datetime.now() - zerotime).total_seconds() * 1000), {SESSION_COUNT: c}))
  ads = adsPerSecondAndType.map(lambda stc: (stc[0][0], {ADS_PER_SEC+"#"+stc[0][1]: stc[1]}))

  #all the streams are unioned and combined
  finalStats = finalSessionCount.union(requests).union(errors).union(ads).reduceByKey(lambda m1, m2: dict(m1.items() + m2.items()))
  def sendMetrics(itr):
      global brokerList
      prod = KafkaProducerWrapper.getProducer([brokerList])
      for m in itr:
          mstr = ",".join([str(x) + "->" + str(m[1][x]) for x in m[1]])
          prod.send(statsTopic, key=str(m[0]), value=str(m[0])+":("+mstr+")")
      prod.flush()

  #Each partitions uses its own Kafka producer (one per partition) to send the formatted message
  finalStats.foreachRDD(lambda rdd: rdd.foreachPartition(sendMetrics))

  print("Starting the streaming context... Kill me with ^C")

  ssc.start()
  ssc.awaitTermination()
```

- 활성 세션 수가 1초를 미니배치로 하기 때문에 초당 타임스탬프를 키로하는 결과를 합쳐서 kafka 전송
- 드라이브에서 초기화한 카프카 프로듀서 객체는 워커로 전송할 수 없다. 그 대신 워커에서 실행되는 태스크에서 프로듀서를 초기화 한다.
    - (스칼라) KafkaProducerWrapper 의 동반 객체는 단일 인스턴스를 lazy instantiation 하며 단일 카프카 프로듀서의 인스턴스를 초기화 한다.
    - foreachPartition 을 사용해 Producer 객체를 JVM 당 하나씩 초기화 하고 메세지를 카프카로 전송한다. (여러 파티션이 동일한 실행자 JVM 을 공유하므로 Producer 객체도 공유 가능?)



## 맵리듀스란?

MapReduce: Simplified Data Processing on Large Clusters

구글의 과제였던 클러스터 컴퓨팅의 단순화 문제를 해결한 방책.

1. 잡을 잘게 분할하고 클러스터의 모든 노드로 매핑해 분산 처리
2. 각 노드는 분할된 잡을 처리한 중간 결과를 생성
3. 분할된 중간 결과를 reduce, 집계해 최종 결과

- 병렬 처리: 잘게 나누어 동시에 처리
- 데이터 분산: 여러 노드로 분산
- 장애 내성: 분산 컴포넌트의 장애에 대응
    - 마스터는 모든 워커 노드에 주기적으로 ping 전송한다. 일정 시간 이상 워커의 응답이 없다면 마스터는 워커에 문제가 발생했다고 간주하여 해당 워커의 모든 맵 태스크를 초기상태로 다른 워커에 스케줄링

⇒ 데이터를 옮겨서 처리하지 않고 데이터가 저장된 곳으로 프로그램을 전송.

**word count**

1. map: 각 문장을 단어로 분할하고 (단어,1) 쌍 목록 반환
2. shuffle phase: 동일한 단어를 동일한 리듀서에 전달 (map 의 결과를 키별로 그루핑해 reduce 에 전달)
    
    → 병목있을 수 있지만 후속 처리 없이 단어 집계함
    
3. reduce: 단어별 출현 횟수를 합산해 최종 결과 생성
