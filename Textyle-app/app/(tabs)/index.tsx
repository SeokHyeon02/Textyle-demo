import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Image, Linking, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { supabase } from '../../supabase';

const FASHION_API_URL = process.env.EXPO_PUBLIC_FASHION_API_URL?.replace(/\/$/, '');
const FASHION_API_V2_URL = FASHION_API_URL?.replace(/:8001(\/|$)/, ':8002$1');

export default function SearchScreen() {
  const [session, setSession] = useState<any>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [searchText, setSearchText] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [loadingVersion, setLoadingVersion] = useState<'v1' | 'v2' | null>(null);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [resultVersion, setResultVersion] = useState<'v1' | 'v2' | null>(null);
  const [resultMeta, setResultMeta] = useState<any>(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    supabase.auth.onAuthStateChange((_event, session) => setSession(session));
  }, []);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 1,
      preferredAssetRepresentationMode: ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Automatic,
    });
    if (!result.canceled) {
      const asset = result.assets[0];
      setImageUri(asset.uri);
      setSelectedImage(asset);
    }
  };

  const searchClothes = async (version: 'v1' | 'v2' = 'v1') => {
    if (!imageUri) {
      Alert.alert('알림', '사진을 선택해주세요!');
      return;
    }

    const baseUrl = version === 'v2' ? FASHION_API_V2_URL : FASHION_API_URL;
    if (!baseUrl) {
      Alert.alert(
        '설정 오류',
        version === 'v2'
          ? 'v2 서버 URL을 만들 수 없습니다. EXPO_PUBLIC_FASHION_API_URL이 :8001 포트를 포함하는지 확인해주세요.'
          : 'EXPO_PUBLIC_FASHION_API_URL 환경변수가 설정되지 않았습니다.',
      );
      return;
    }

    setIsLoading(true);
    setLoadingVersion(version);

    try {
      const formData = new FormData();
      const uploadUri = selectedImage?.uri || imageUri;
      const fileName = selectedImage?.fileName || 'photo.jpg';
      const mimeType = selectedImage?.mimeType || 'image/jpeg';

      formData.append('file', {
        uri: uploadUri,
        name: fileName,
        type: mimeType,
      } as any);

      formData.append('query', searchText.trim());

      const response = await fetch(`${baseUrl}/search`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '서버 오류');
      }
      const data = await response.json();
      setSearchResults(data.results);
      setResultVersion(version);
      setResultMeta({
        enhanced_query: data.enhanced_query,
        design_description: data.design_description,
        color_extracted: data.color_extracted,
        intent: data.intent,
      });
    } catch (error) {
      console.error("검색 에러:", error);
      Alert.alert('통신 에러', '서버에 연결할 수 없습니다. 파이썬 서버가 켜져 있는지, IP 주소가 맞는지 확인해주세요.');
    } finally {
      setIsLoading(false);
      setLoadingVersion(null);
    }
  };

  // ⭐️ 새로 추가된 안전장치 1: 링크를 무조건 열리게 포장해주는 함수
  const openShopLink = async (link: string) => {
    if (!link) {
      Alert.alert('알림', '상품 링크가 없습니다.');
      return;
    }
    let targetUrl = link.trim();
    // 링크가 http로 안 시작하면 강제로 붙여버리기!
    if (targetUrl.startsWith('//')) {
      targetUrl = 'https:' + targetUrl;
    } else if (!targetUrl.startsWith('http')) {
      targetUrl = 'https://' + targetUrl;
    }

    try {
      await Linking.openURL(targetUrl);
    } catch (e) {
      Alert.alert('오류', '링크를 열 수 없습니다.');
    }
  };

  // ⭐️ 새로 추가된 안전장치 2: 사진 URL을 앱이 좋아하는 형태로 다듬는 함수
  const getValidImageUrl = (url: string) => {
    if (!url) return 'https://via.placeholder.com/90?text=No+Image';
    let validUrl = url.trim();
    if (validUrl.startsWith('//')) {
      validUrl = 'https:' + validUrl;
    }
    return validUrl;
  };

  if (!session) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centerContainer}>
          <Text style={styles.title}>TexTyle AI</Text>
          <Text style={styles.subtitle}>스마트한 패션 검색을 시작해보세요</Text>
          <TouchableOpacity style={styles.loginButton} onPress={() => router.push('/login')}>
            <Text style={styles.loginButtonText}>로그인하고 검색하기</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (searchResults.length > 0) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.resultContainer}>
          <View style={styles.resultHeader}>
            <Text style={styles.searchTitle}>✨ 찰떡같은 옷을 찾았어요!</Text>
            {resultVersion && (
              <View style={[styles.versionBadge, resultVersion === 'v2' ? styles.versionBadgeV2 : styles.versionBadgeV1]}>
                <Text style={styles.versionBadgeText}>
                  {resultVersion === 'v2' ? 'v2 · 텍스트 전용' : 'v1 · 이미지+텍스트'}
                </Text>
              </View>
            )}
          </View>

          {resultMeta?.enhanced_query && (
            <View style={styles.metaBox}>
              <Text style={styles.metaLabel}>검색 쿼리</Text>
              <Text style={styles.metaValue}>{resultMeta.enhanced_query}</Text>
              {resultMeta.color_extracted?.color && (
                <>
                  <Text style={styles.metaLabel}>추출된 색상</Text>
                  <Text style={styles.metaValue}>
                    {resultMeta.color_extracted.color} ({resultMeta.color_extracted.confidence})
                    {resultMeta.color_extracted.pattern ? ` · ${resultMeta.color_extracted.pattern}` : ''}
                  </Text>
                </>
              )}
              {resultMeta.design_description && (
                <>
                  <Text style={styles.metaLabel}>디자인 설명</Text>
                  <Text style={styles.metaValue}>{resultMeta.design_description}</Text>
                </>
              )}
            </View>
          )}

          {searchResults.map((item, index) => (
            <View key={index} style={styles.resultCard}>
              {/* 🚨 기존 headers 다 빼고, 안전장치 함수만 통과시켰습니다! */}
              <Image
                source={{ uri: getValidImageUrl(item.image_url) }}
                style={styles.resultImage}
                resizeMode="cover"
              />
              <View style={styles.resultInfo}>
                {/* ⭐️ 변경된 부분: 대분류 > 소분류 카테고리 */}
                <Text style={styles.resultCategory}>[{item.main_category} {' > '} {item.sub_category}]</Text>
                <Text style={styles.resultBrand}>{item.brand_name}</Text>
                <Text style={styles.resultName} numberOfLines={2}>{item.name}</Text>

                {/* ⭐️ 새로 추가된 부분: 천 단위 콤마 가격 표시 */}
                <Text style={styles.resultPrice}>
                  {item.price ? `${Number(item.price).toLocaleString()}원` : '가격 정보 없음'}
                </Text>

                <Text style={styles.resultSimilarity}>일치율: {(item.similarity * 100).toFixed(1)}%</Text>

                {/* 🚨 링크 열기에도 안전장치 함수를 달았습니다! */}
                <TouchableOpacity onPress={() => openShopLink(item.shop_link)}>
                  <Text style={styles.resultLink}>무신사에서 보기 🔗</Text>
                </TouchableOpacity>
              </View>
            </View>
          ))}

          <TouchableOpacity
            style={styles.resetButton}
            onPress={() => {
              setSearchResults([]);
              setResultVersion(null);
              setResultMeta(null);
            }}
          >
            <Text style={styles.resetButtonText}>다른 옷 검색하기</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <View style={styles.mainContent}>
          <Text style={styles.searchTitle}>무엇을 찾고 계신가요?</Text>

          <TextInput
            style={styles.textInput}
            placeholder="예) 이 사진과 색깔이 비슷한 의류를 찾아줘"
            value={searchText}
            onChangeText={setSearchText}
          />

          <TouchableOpacity style={styles.imageContainer} onPress={pickImage}>
            {imageUri ? (
              <Image source={{ uri: imageUri }} style={styles.image} />
            ) : (
              <View style={styles.imagePlaceholder}>
                <Text style={styles.placeholderIcon}>📷</Text>
                <Text style={styles.placeholderText}>옷 사진 첨부하기 (클릭)</Text>
              </View>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.searchButton, isLoading && styles.searchButtonDisabled]}
            onPress={() => searchClothes('v1')}
            disabled={isLoading}
          >
            {isLoading && loadingVersion === 'v1' ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.searchButtonText}>v1 검색 (이미지+텍스트) 🔍</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.searchButtonV2, isLoading && styles.searchButtonDisabled]}
            onPress={() => searchClothes('v2')}
            disabled={isLoading}
          >
            {isLoading && loadingVersion === 'v2' ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.searchButtonText}>v2 검색 (텍스트 전용) 🧪</Text>
            )}
          </TouchableOpacity>
        </View>

        <View style={styles.adBanner}>
          <Text style={styles.adText}>광고 배너가 들어갈 자리입니다</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

// 스타일링
const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#fff' },
  container: { flex: 1, paddingHorizontal: 20 },
  centerContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  mainContent: { flex: 1, justifyContent: 'center' },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 10, color: '#333' },
  searchTitle: { fontSize: 22, fontWeight: 'bold', marginBottom: 20, color: '#333', textAlign: 'center' },
  subtitle: { fontSize: 16, color: '#666', marginBottom: 30 },
  loginButton: { backgroundColor: '#8B5CF6', paddingVertical: 15, paddingHorizontal: 30, borderRadius: 25 },
  loginButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  textInput: { height: 50, borderColor: '#ddd', borderWidth: 1, borderRadius: 10, paddingHorizontal: 15, marginBottom: 20, fontSize: 16, backgroundColor: '#FAFAFA' },
  imageContainer: { height: 250, backgroundColor: '#f9f9f9', borderRadius: 15, borderWidth: 1.5, borderColor: '#ddd', borderStyle: 'dashed', overflow: 'hidden', marginBottom: 20, justifyContent: 'center', alignItems: 'center' },
  imagePlaceholder: { alignItems: 'center' },
  placeholderIcon: { fontSize: 40, marginBottom: 10 },
  placeholderText: { color: '#888', fontSize: 16 },
  image: { width: '100%', height: '100%' },
  searchButton: { backgroundColor: '#8B5CF6', height: 55, borderRadius: 10, justifyContent: 'center', alignItems: 'center' },
  searchButtonV2: { backgroundColor: '#10B981', height: 55, borderRadius: 10, justifyContent: 'center', alignItems: 'center', marginTop: 10 },
  searchButtonDisabled: { opacity: 0.6 },
  searchButtonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  resultHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  versionBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  versionBadgeV1: { backgroundColor: '#8B5CF6' },
  versionBadgeV2: { backgroundColor: '#10B981' },
  versionBadgeText: { color: '#fff', fontSize: 12, fontWeight: 'bold' },
  metaBox: { backgroundColor: '#F9FAFB', borderRadius: 10, padding: 12, marginBottom: 15, borderWidth: 1, borderColor: '#E5E7EB' },
  metaLabel: { fontSize: 11, color: '#6B7280', fontWeight: 'bold', marginTop: 4, textTransform: 'uppercase' },
  metaValue: { fontSize: 13, color: '#374151', marginTop: 2 },
  adBanner: { height: 60, backgroundColor: '#F3F4F6', justifyContent: 'center', alignItems: 'center', borderRadius: 8, marginBottom: 10 },
  adText: { color: '#9CA3AF', fontSize: 14 },
  resultContainer: { flex: 1, padding: 20 },
  resultCard: { flexDirection: 'row', backgroundColor: '#FAFAFA', borderRadius: 12, padding: 12, marginBottom: 15, borderWidth: 1, borderColor: '#EEE' },
  resultImage: { width: 90, height: 90, borderRadius: 8, marginRight: 15 },
  resultInfo: { flex: 1, justifyContent: 'center' },
  resultCategory: { fontSize: 12, color: '#8B5CF6', fontWeight: 'bold', marginBottom: 4 },
  resultName: { fontSize: 15, fontWeight: '600', color: '#333', marginBottom: 6 },

  // ⭐️ 새로 추가된 가격 텍스트 스타일
  resultPrice: { fontSize: 16, fontWeight: 'bold', color: '#333', marginTop: 2, marginBottom: 4 },

  resultSimilarity: { fontSize: 13, color: '#10B981', marginBottom: 6, fontWeight: 'bold' },
  resultLink: { fontSize: 14, color: '#3B82F6', textDecorationLine: 'underline' },
  resetButton: { backgroundColor: '#333', height: 50, borderRadius: 10, justifyContent: 'center', alignItems: 'center', marginTop: 10, marginBottom: 40 },
  resetButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  resultBrand: {
    fontSize: 13, color: '#333', fontWeight: '600', marginBottom: 2,
  },
});
