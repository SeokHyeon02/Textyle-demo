import { Ionicons } from '@expo/vector-icons';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const SEARCH_EXAMPLES = [
  '같은 색 와이드 청바지',
  '검정 말고 다른 색 청바지',
  '회색 와이드 팬츠',
  '이 디자인과 색상이 비슷한 바지',
];

function GuideSection({
  icon,
  title,
  body,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  body: string;
}) {
  return (
    <View style={styles.section}>
      <View style={styles.sectionHeader}>
        <View style={styles.iconBadge}>
          <Ionicons name={icon} size={18} color="#3E6AE1" />
        </View>
        <Text style={styles.sectionTitle}>{title}</Text>
      </View>
      <Text style={styles.sectionBody}>{body}</Text>
    </View>
  );
}

export default function GuideScreen() {
  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.headerBlock}>
          <Text style={styles.screenTitle}>사용 안내</Text>
          <Text style={styles.subtitle}>
            사진과 검색어를 함께 사용해{'\n'}
            원하는 옷과 비슷한 상품을 찾을 수 있습니다.
          </Text>
        </View>

        <GuideSection
          icon="image-outline"
          title="1. 찾고 싶은 옷 사진을 선택하세요"
          body="검색 기준이 되는 옷이 잘 보이는 사진을 고르세요. 사진만으로도 비슷한 상품을 찾을 수 있지만, 검색어를 함께 쓰면 조건을 더 정확히 반영할 수 있습니다."
        />

        <GuideSection
          icon="chatbubble-ellipses-outline"
          title="2. 원하는 조건을 검색어로 적으세요"
          body="색상, 핏, 소재, 제외하고 싶은 색을 자연스럽게 입력하면 됩니다. 예를 들어 '같은 색 와이드 청바지'는 사진과 비슷한 색의 와이드 청바지를 찾고, '검정 말고 다른 색 청바지'는 검정색을 피해서 찾습니다."
        />

        <View style={styles.exampleBox}>
          <Text style={styles.exampleTitle}>검색어 예시</Text>
          <View style={styles.exampleList}>
            {SEARCH_EXAMPLES.map(example => (
              <Text key={example} style={styles.examplePill}>{example}</Text>
            ))}
          </View>
        </View>

        <GuideSection
          icon="scan-outline"
          title="3. 필요한 경우에만 정밀 분석을 켜세요"
          body="GroundingDINO 정밀 분석은 옷 영역을 먼저 분리해 색상 분석에 사용합니다. 배경이 복잡하거나 사진 안에 여러 물체가 있을 때만 켜는 것을 권장합니다. 켜면 검색 시간이 길어질 수 있습니다."
        />

        <GuideSection
          icon="analytics-outline"
          title="4. 결과의 판단 근거를 확인하세요"
          body="결과 화면에는 서버가 해석한 검색어, 추출한 이미지 색상, 카테고리, 이미지 유사도, 종합 점수가 표시됩니다. 색상 일치나 카테고리 일치 같은 라벨을 보면 왜 해당 상품이 추천됐는지 확인할 수 있습니다."
        />

        <View style={styles.tipBox}>
          <Ionicons name="bulb-outline" size={18} color="#755100" />
          <Text style={styles.tipText}>
            결과가 마음에 들지 않으면 검색어를 더 구체적으로 바꿔보세요. 색상, 핏, 소재를 함께 적을수록 원하는 결과에 가까워집니다.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  content: {
    paddingHorizontal: 20,
    paddingTop: 28,
    paddingBottom: 32,
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
  },
  headerBlock: {
    width: '100%',
    maxWidth: 430,
    alignItems: 'center',
    marginBottom: 20,
  },
  screenTitle: {
    fontSize: 26,
    fontWeight: '600',
    color: '#171A20',
    textAlign: 'center',
  },
  subtitle: {
    marginTop: 8,
    fontSize: 15,
    lineHeight: 22,
    color: '#393C41',
    textAlign: 'center',
  },
  section: {
    width: '100%',
    maxWidth: 430,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#FFFFFF',
    padding: 16,
    marginTop: 12,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  iconBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#F0F4FF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sectionTitle: {
    flex: 1,
    fontSize: 17,
    lineHeight: 23,
    fontWeight: '600',
    color: '#171A20',
  },
  sectionBody: {
    marginTop: 10,
    fontSize: 14,
    lineHeight: 21,
    color: '#393C41',
  },
  exampleBox: {
    width: '100%',
    maxWidth: 430,
    marginTop: 12,
    borderRadius: 8,
    backgroundColor: '#F4F4F4',
    padding: 14,
  },
  exampleTitle: {
    fontSize: 14,
    lineHeight: 20,
    fontWeight: '600',
    color: '#171A20',
  },
  exampleList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 10,
  },
  examplePill: {
    borderRadius: 6,
    backgroundColor: '#FFFFFF',
    color: '#393C41',
    fontSize: 12,
    lineHeight: 17,
    paddingHorizontal: 9,
    paddingVertical: 6,
  },
  tipBox: {
    width: '100%',
    maxWidth: 430,
    marginTop: 14,
    borderRadius: 8,
    backgroundColor: '#FFF8E8',
    padding: 14,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
  },
  tipText: {
    flex: 1,
    fontSize: 13,
    lineHeight: 19,
    color: '#5F4200',
  },
});
