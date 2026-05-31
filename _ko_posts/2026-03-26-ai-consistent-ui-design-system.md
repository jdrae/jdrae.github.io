---
title: "디자인 지식이 없는 개발자가 AI로 일관된 UI를 만드는 방법"
date: 2026-03-26
permalink: /ko/2026/03/26/ai-consistent-ui-design-system/
categories: [agentic-coding]
tags: [figma, design-system]
translation_key: ai-consistent-ui-design-system
---

웹이나 앱 개발을 하면서 가장 어렵게 느껴지는 부분이 무엇이냐고 묻는다면, 저는 가장 먼저 디자인을 떠올립니다.

예전 같았다면 “디자인보다는 기능이 더 중요하지 않나?”라고 생각했을지도 모르겠습니다. 하지만 실제로 사용자들의 눈길을 먼저 끄는 것은 대개 “잘 만든 기능”보다 “예쁘게 보이는 화면”인 경우가 많습니다. 물론 기능이 중요하지 않다는 뜻은 아닙니다. 다만 사용자가 처음 마주하는 것은 코드가 아니라 화면이라는 점을 인정할 수밖에 없습니다.

게다가 요즘은 디자이너분들의 바이브 코딩 실력도 점점 좋아지고 있습니다. 예쁘면서 기능까지 잘 돌아가는 서비스가 정말 많아졌습니다. 결국 개발자도 어느 정도는 보기 좋은 제품을 만들 수 있어야 하거나, 디자인이 조금 부족하더라도 기능적으로 확실히 차별화된 제품을 만들어야 하는 시대가 된 것 같습니다.

어쨌든 예쁘게 만든다고 해서 손해 볼 일은 없으니, 저도 디자인을 조금은 알아두면 좋겠다고 생각했습니다.

## 어떻게 하면 디자인을 잘 할.. 아니 잘 시킬 수 있을까?

그렇다고 디자인을 처음부터 정석으로 공부하는 것은 제 입장에서 너무 비효율적으로 느껴졌습니다. 대 에이전트의 시대에 “내가 직접 디자인을 잘하는 것”보다 “AI에게 디자인을 잘 시키는 방법”을 먼저 찾아보기로 했습니다.

우선 Figma AI와 Stitch를 사용해봤지만.. 단순히 프롬프트만 입력한다고 해서 한 번에 만족스러운 결과가 나오지는 않았습니다. 레이아웃 아이디어를 얻는 용도로는 괜찮았지만, 실제 서비스에 바로 적용할 수 있을 정도의 결과물을 얻기는 쉽지 않았습니다. (물론 제가 디자인 프롬프트를 잘 작성하지 못해서 그랬을 수도 있습니다)

그래서 유튜브에서 여러 디자인 방법론을 찾아보고, 실제 디자이너들이 AI 에이전트를 어떻게 활용하는지도 살펴봤습니다.

그러다가 알게 된 것이 바로 디자인 시스템(Design System) 이었습니다.

## 문제는 ‘감성’이 아니라 ‘구체화’

디자인 시스템을 접하고 나서야 제가 헤매고 있던 이유를 조금 알 것 같았습니다. 문제는 제 디자인 감성이 부족해서라기보다, 머릿속에 있는 추상적인 이미지를 구체적인 규칙으로 정리하지 못했기 때문이었습니다.

디자인 시스템은 서비스에서 사용할 디자인 요소를 미리 정의해두고, 이를 일관되게 사용할 수 있도록 돕는 체계입니다. 예를 들어 색상, 폰트 크기, 간격, 버튼 스타일, 카드 형태, 모달 디자인 등을 미리 정해두는 것입니다.

AI 에이전트는 요청할 때마다 조금씩 다른 디자인을 만들어내는 경우가 많습니다. 그래서 매번 “예쁘게 해줘”라고 요청하는 대신, 미리 정의된 디자인 시스템을 참고해서 작업하게 하면 훨씬 더 일관된 결과를 만들어 준다는 정보를 봤습니다.

## 디자인 시스템은 UX를 도와주지 않는다

단, 저는 화면 기획 자체는 직접 했습니다.

손으로 먼저 와이어프레임을 그리고, 이후 Figma로 옮겨두었습니다. 이 부분은 UX와 관련된 영역이기 때문에 AI에게 전부 맡기기보다는, 사용자의 흐름과 화면 구조를 직접 고민하는 편이 낫다고 판단했습니다.

디자인 시스템은 UI 구현을 편리하게 만들어주는 도구에 가깝습니다. 디자인 시스템을 만든다고 해서 아무런 기획 없이 완벽한 UI/UX가 자동으로 튀어나오지는 않습니다. 맨땅에 헤딩하는 것보다는 훨씬 낫지만, 그래도 어디로 헤딩할지는 정해야 합니다.

## 디자인 시스템을 만들고 적용하기

### 1. 레퍼런스 탐색하기

먼저 만들고 싶은 앱의 분위기를 정했습니다.

최소한 아래 정도는 정해두는 것이 좋습니다.

- 앱의 전체 콘셉트
    - 예: 심플, 아날로그, 다크, 미니멀, 감성적인 분위기 등
- 메인 컬러
- 보조 컬러 1~2개

레퍼런스는 Pinterest에서 수집해도 좋고, 따라 하고 싶은 웹사이트나 앱을 찾아봐도 좋습니다. 중요한 것은 “이런 느낌이면 좋겠다”를 눈으로 확인할 수 있는 자료를 모아두는 것입니다.

AI에게도 참고할 기준이 있어야 합니다. 사람에게 “알아서 예쁘게 해주세요”라고 하면 난감한 것처럼, AI도 제대로 알려주지 않으면 제멋대로 만들게 됩니다.

![디자인 레퍼런스 예시](/assets/images/posts/2026-03-26-ai-consistent-ui-design-system/ai-consistent-ui-reference.png)

### 2. Figma MCP로 디자인 시스템 만들기

레퍼런스와 콘셉트를 정한 뒤, Figma MCP와 연결해서 디자인 시스템을 만들어달라고 요청했습니다.

이때 Figma에서 결과물을 확인하면서 계속 수정했습니다. 추가로 버튼, 모달, 카드 같은 기본 컴포넌트도 미리 만들어두면 이후 작업이 훨씬 편해집니다.

디자인 시스템 프롬프트는 `design system prompt`로 검색하면 여러 예시를 찾을 수 있습니다. 저는 당시 검색해서 바로 나온 GitHub 프롬프트를 사용했습니다.

[Design System Foundation Prompt](https://github.com/dfolloni82/design-system-prompts/blob/main/1-design-system-foundation.md#design-summary-also-provide)

디자인 시스템을 만들 때는 최소한 아래 항목들이 포함되어 있는지 확인하는 것이 좋습니다.

- Color Token
- Spacing Scale
- Typography Scale
- Radius
- Shadow / Elevation
- 기본 컴포넌트
    - Button
    - Input
    - Card
    - Modal
    - List Item
- 각 컴포넌트의 상태
    - Default
    - Pressed
    - Disabled
    - Error
    - Loading

![Figma에 정리한 디자인 시스템 컴포넌트](/assets/images/posts/2026-03-26-ai-consistent-ui-design-system/design-system-figma-components.png)

### 3. 프론트엔드 프로젝트에 디자인 시스템 코드화하기

다음으로 프론트엔드 프로젝트 폴더에서 Figma MCP를 연결하고, Figma에 구현된 디자인 시스템을 기준으로 코드를 작성해달라고 요청했습니다.

이때 중요한 점은 색상, 폰트 크기, 간격 등을 하드코딩하지 않도록 하는 것입니다.

예를 들어 특정 버튼마다 `#3366FF` 같은 색상 값을 직접 넣어두면, 나중에 메인 컬러를 바꾸고 싶을 때 모든 파일을 뒤져야 합니다. 대신 디자인 토큰 형태로 관리하면 한 번에 수정할 수 있습니다.

저는 추가로 디자인 시스템을 글로 정의한 Markdown 파일도 만들어달라고 요청했습니다. 이렇게 해두면 이후 AI 에이전트에게 작업을 맡길 때도 “이 문서를 기준으로 구현해줘”라고 명확하게 지시할 수 있습니다.

![Markdown으로 정리한 디자인 시스템 문서](/assets/images/posts/2026-03-26-ai-consistent-ui-design-system/design-system-markdown.png)

### 4. 실제 프론트엔드 화면에 적용하기

마지막으로 코드로 구현된 디자인 시스템을 실제 프론트엔드 화면에 적용했습니다.

이후 화면 디자인을 수정할 때마다 디자인 시스템에 맞는지 계속 확인했습니다. 저는 이미 Figma에서 직접 기획한 앱 화면들이 있었고, 대부분의 구조는 정해져 있었습니다. 그래서 spacing, font size, radius 같은 세부 값만 조정하면 되는 상황이었습니다.

이 방식의 장점은 “매번 새로 디자인하는 느낌”이 아니라 “정해진 규칙 안에서 다듬는 느낌”으로 작업할 수 있다는 점입니다. AI에게도 훨씬 명확하게 지시할 수 있고, 결과물도 덜 들쭉날쭉해집니다.

![디자인 시스템 적용 전 화면](/assets/images/posts/2026-03-26-ai-consistent-ui-design-system/design-system-before-application.png)

![디자인 시스템 적용 후 화면](/assets/images/posts/2026-03-26-ai-consistent-ui-design-system/design-system-after-application.png)


## 추가: 랜딩페이지를 만들 때 활용한 방법

개발자는 디자인 지식이 부족한 경우가 많기 때문에, 자신이 원하는 디자인이 정확히 어떤 스타일인지 설명하는 것도 쉽지 않습니다.

아래 Reddit 글에서 소개된 프롬프트는 25가지의 서로 다른 디자인 스타일을 제공하고, 그중 랜덤으로 하나를 선택해 해당 스타일에 대한 세부 프롬프트를 다시 작성한 뒤, 최종적으로 랜딩페이지를 만들게 하는 방식입니다.

[Reddit - I made a prompt to generate unique beautiful landing pages](https://www.reddit.com/r/PromptEngineering/comments/1p5ztk4/i_made_a_prompt_to_generate_unique_beautiful/)

다만 랜덤 방식으로 진행하면 원하는 디자인이 나올 때까지 여러 번 시도해야 한다는 단점이 있습니다.

그래서 저는 먼저 위의 프롬프트에 정의된 디자인 스타일 설명을 읽고, 마음에 드는 디자인 철학을 하나 고른 뒤, 그 디자인 철학에 대한 세부 프롬프트를 작성해달라고 요청했습니다.

즉, 바로 랜딩페이지를 만들게 하기보다는 디자인을 더 구체화하는 메타 프롬프팅 과정을 먼저 거친 것입니다. 이렇게 몇 번 시도해보면 원하는 결과에 가까운 랜딩페이지를 얻을 수 있었습니다.

![랜딩페이지 디자인 스타일 프롬프트 예시](/assets/images/posts/2026-03-26-ai-consistent-ui-design-system/landing-page-style-prompt.png)

## 추천 자료

- [I Built My Entire Design System in 4 Hours With AI. Full Tutorial (Claude + Cursor + Figma)](https://www.youtube.com/watch?v=nafNPuElCtY&t=278s)
- [How I Build a Production-Ready Design System with AI](https://www.youtube.com/watch?v=sTNy6cDMEjg&t=235s)

## 흥미롭게 본 영상

- [Design Systems are a Waste of Time Now](https://www.youtube.com/watch?v=6_t66Ef0Llk&t=26s)
