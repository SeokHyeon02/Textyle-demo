import { Ionicons } from '@expo/vector-icons';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function BookmarksScreen() {
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <View style={styles.emptyPanel}>
          <Ionicons name="heart-outline" size={42} color="#5C5E62" />
          <Text style={styles.title}>찜한 상품</Text>
          <Text style={styles.body}>마음에 드는 상품을 저장하는 기능을 준비 중입니다.</Text>
          <Text style={styles.helper}>지금은 검색 결과에서 상품 링크로 바로 이동할 수 있어요.</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  container: {
    flex: 1,
    paddingHorizontal: 20,
    justifyContent: 'center',
    backgroundColor: '#FFFFFF',
  },
  emptyPanel: {
    minHeight: 260,
    borderRadius: 8,
    backgroundColor: '#F4F4F4',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  title: {
    marginTop: 16,
    fontSize: 24,
    fontWeight: '600',
    color: '#171A20',
    textAlign: 'center',
  },
  body: {
    marginTop: 10,
    fontSize: 15,
    lineHeight: 22,
    color: '#393C41',
    textAlign: 'center',
  },
  helper: {
    marginTop: 8,
    fontSize: 13,
    lineHeight: 19,
    color: '#5C5E62',
    textAlign: 'center',
  },
});
