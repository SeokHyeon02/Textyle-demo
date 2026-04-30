import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Image, Linking, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { supabase } from '../../supabase';

export default function SearchScreen() {
  const [session, setSession] = useState<any>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [searchText, setSearchText] = useState('');
  
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    supabase.auth.onAuthStateChange((_event, session) => setSession(session));
  }, []);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 5],
      quality: 0.8,
    });
    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    }
  };

  const searchClothes = async () => {
    if (!imageUri || !searchText.trim()) {
      Alert.alert('알림', '사진과 검색어를 모두 입력해주세요!');
      return;
    }

    setIsLoading(true);

    try {
      const formData = new FormData();
      const uriParts = imageUri.split('.');
      const fileType = uriParts[uriParts.length - 1];
      
      formData.append('file', {
        uri: imageUri,
        name: `photo.${fileType}`,
        type: `image/${fileType}`,
      } as any);

      formData.append('query', searchText.trim());

      const SERVER_IP = "192.168.0.6"; // 🚨 본인 IP 확인!
      const response = await fetch(`http://${SERVER_IP}:8001/search`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '서버 오류');
      }
      const data = await response.json();
      setSearchResults(data.results);
    } catch (error) {
      console.error("검색 에러:", error);
      Alert.alert('통신 에러', '서버에 연결할 수 없습니다. 파이썬 서버가 켜져 있는지, IP 주소가 맞는지 확인해주세요.');
    } finally {
      setIsLoading(false);
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
          <Text style={styles.searchTitle}>✨ 찰떡같은 옷을 찾았어요!</Text>
          
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

          <TouchableOpacity style={styles.resetButton} onPress={() => setSearchResults([])}>
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

          <TouchableOpacity style={styles.searchButton} onPress={searchClothes} disabled={isLoading}>
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.searchButtonText}>비슷한 옷 찾기 🔍</Text>
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
  searchButtonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
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