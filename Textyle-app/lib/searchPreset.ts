// 검색 탭에 "미리 첨부할 이미지"를 한 번만 전달하기 위한 일회성 저장소.
// 상품 상세 페이지의 '검색하기' → 검색 탭으로 이동할 때 사용한다.
let presetImageUrl: string | null = null;

export function setSearchPresetImage(url: string | null) {
  presetImageUrl = url;
}

// 값을 읽고 즉시 비운다(일회성). 같은 이미지가 중복으로 다시 첨부되는 것을 막는다.
export function consumeSearchPresetImage(): string | null {
  const value = presetImageUrl;
  presetImageUrl = null;
  return value;
}
