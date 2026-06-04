import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import type { Session } from '@supabase/supabase-js';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Linking,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { supabase } from '../../supabase';

const FASHION_API_URL = process.env.EXPO_PUBLIC_FASHION_API_URL?.replace(/\/$/, '');
// v3 서버는 포트만 8001 → 8003 으로 치환 (별도 env 불필요)
const FASHION_API_V3_URL = FASHION_API_URL?.replace(/:8001(\/|$)/, ':8003$1');
const PLACEHOLDER_IMAGE_URL = 'https://via.placeholder.com/200?text=No+Image';

type SearchVersion = 'v1' | 'v3';

type SearchResult = {
  image_url?: string | null;
  main_category?: string | null;
  sub_category?: string | null;
  brand_name?: string | null;
  name?: string | null;
  price?: number | string | null;
  similarity?: number | null;
  shop_link?: string | null;
};

export default function SearchScreen() {
  const [session, setSession] = useState<Session | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [searchText, setSearchText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingVersion, setLoadingVersion] = useState<SearchVersion | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [resultVersion, setResultVersion] = useState<SearchVersion | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // v3 테스트 노브 (비우면 서버 기본값 사용)
  const [imageWeightInput, setImageWeightInput] = useState('');     // w2
  const [scoreThresholdInput, setScoreThresholdInput] = useState(''); // 하드컷
  const [colorPenaltyInput, setColorPenaltyInput] = useState('');   // 색 페널티 배율
  const [changeGate, setChangeGate] = useState(true);               // change 쿼리에서 w2=0

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    const { data } = supabase.auth.onAuthStateChange((_event, session) => setSession(session));

    return () => data.subscription.unsubscribe();
  }, []);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 1,
      preferredAssetRepresentationMode: ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Automatic,
    });

    if (!result.canceled) {
      const asset = result.assets[0];
      setImageUri(asset.uri);
      setSelectedImage(asset);
      setErrorMessage(null);
    }
  };

  const parseKnob = (value: string): number | null => {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  };

  const searchClothes = async (version: SearchVersion = 'v1') => {
    if (!imageUri) {
      setErrorMessage('검색할 사진을 먼저 선택해주세요.');
      return;
    }

    const baseUrl = version === 'v3' ? FASHION_API_V3_URL : FASHION_API_URL;
    if (!baseUrl) {
      setErrorMessage(
        version === 'v3'
          ? 'v3 서버 URL을 만들 수 없습니다. EXPO_PUBLIC_FASHION_API_URL이 :8001 포트를 포함하는지 확인해주세요.'
          : 'EXPO_PUBLIC_FASHION_API_URL 환경변수가 설정되지 않았습니다.',
      );
      return;
    }

    setIsLoading(true);
    setLoadingVersion(version);
    setErrorMessage(null);

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

      if (version === 'v3') {
        const iw = parseKnob(imageWeightInput);
        const st = parseKnob(scoreThresholdInput);
        const cp = parseKnob(colorPenaltyInput);
        if (iw !== null) formData.append('image_weight', String(iw));
        if (st !== null) formData.append('score_threshold', String(st));
        if (cp !== null) formData.append('color_penalty_scale', String(cp));
        formData.append('change_gate', changeGate ? 'true' : 'false');
      }

      const response = await fetch(`${baseUrl}/search`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || '검색 요청을 처리하지 못했습니다.');
      }

      const data = await response.json();
      setSearchResults(Array.isArray(data.results) ? data.results : []);
      setResultVersion(version);
      setHasSearched(true);
    } catch (error) {
      console.error('검색 에러:', error);
      const message = error instanceof Error
        ? error.message
        : '서버에 연결할 수 없습니다. FastAPI 서버와 API 주소를 확인해주세요.';
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
      setLoadingVersion(null);
    }
  };

  const openShopLink = async (link?: string | null) => {
    if (!link) {
      Alert.alert('알림', '상품 링크가 없습니다.');
      return;
    }

    let targetUrl = link.trim();
    if (targetUrl.startsWith('//')) {
      targetUrl = 'https:' + targetUrl;
    } else if (!targetUrl.startsWith('http')) {
      targetUrl = 'https://' + targetUrl;
    }

    try {
      await Linking.openURL(targetUrl);
    } catch {
      Alert.alert('오류', '링크를 열 수 없습니다.');
    }
  };

  const getValidImageUrl = (url?: string | null) => {
    if (!url) return PLACEHOLDER_IMAGE_URL;
    const validUrl = url.trim();
    if (validUrl.startsWith('//')) {
      return 'https:' + validUrl;
    }
    return validUrl;
  };

  const formatCategory = (item: SearchResult) => {
    const parts = [item.main_category, item.sub_category].filter(Boolean);
    return parts.length > 0 ? parts.join(' > ') : '카테고리 정보 없음';
  };

  const formatPrice = (price?: number | string | null) => {
    if (price === null || price === undefined || price === '') return '가격 정보 없음';
    const numericPrice = Number(price);
    return Number.isFinite(numericPrice) ? `${numericPrice.toLocaleString()}원` : String(price);
  };

  const formatSimilarity = (similarity?: number | null) => {
    if (typeof similarity !== 'number') return null;
    return `유사도 ${(similarity * 100).toFixed(1)}%`;
  };

  const resetResults = () => {
    setSearchResults([]);
    setHasSearched(false);
    setErrorMessage(null);
  };

  if (!session) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centerContainer}>
          <Text style={styles.brandTitle}>Textyle</Text>
          <Text style={styles.subtitle}>이미지와 문장으로 비슷한 옷을 찾아보세요.</Text>
          <TouchableOpacity style={styles.primaryButton} onPress={() => router.push('/login')}>
            <Text style={styles.primaryButtonText}>로그인하고 검색하기</Text>
          </TouchableOpacity>
          <Text style={styles.helperText}>가입 후 이미지 검색과 결과 확인을 사용할 수 있습니다.</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (hasSearched) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.resultContent}>
          <View style={styles.resultHeader}>
            <Text style={styles.screenTitle}>검색 결과</Text>
            <View style={styles.resultHeaderRight}>
              {resultVersion && (
                <View style={[styles.versionBadge, resultVersion === 'v3' ? styles.versionBadgeV3 : styles.versionBadgeV1]}>
                  <Text style={styles.versionBadgeText}>
                    {resultVersion === 'v3' ? 'v3 하이브리드' : 'v1 기본'}
                  </Text>
                </View>
              )}
              <Text style={styles.resultCount}>{searchResults.length}개 상품</Text>
            </View>
          </View>

          {searchResults.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="search-outline" size={40} color="#8E8E8E" />
              <Text style={styles.emptyTitle}>조건에 맞는 상품을 찾지 못했어요.</Text>
              <Text style={styles.emptyBody}>다른 사진이나 더 넓은 조건으로 다시 검색해보세요.</Text>
              <TouchableOpacity style={styles.primaryButton} onPress={resetResults}>
                <Text style={styles.primaryButtonText}>검색 화면으로 돌아가기</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <>
              {searchResults.map((item, index) => {
                const similarityText = formatSimilarity(item.similarity);

                return (
                  <View key={`${item.name ?? 'result'}-${index}`} style={styles.resultCard}>
                    <Image
                      source={{ uri: getValidImageUrl(item.image_url) }}
                      style={styles.resultImage}
                      resizeMode="cover"
                    />
                    <View style={styles.resultInfo}>
                      <View style={styles.infoSection}>
                        <Text style={styles.infoLabel}>브랜드</Text>
                        <Text style={styles.resultBrand} numberOfLines={1}>
                          {item.brand_name || '브랜드 정보 없음'}
                        </Text>
                      </View>
                      <View style={styles.infoDivider} />
                      <View style={styles.infoSection}>
                        <Text style={styles.infoLabel}>제품명</Text>
                        <Text style={styles.resultName} numberOfLines={2}>
                          {item.name || '상품명 정보 없음'}
                        </Text>
                      </View>
                      <View style={styles.infoDivider} />
                      <View style={styles.infoGrid}>
                        <View style={styles.infoGridItem}>
                          <Text style={styles.infoLabel}>카테고리</Text>
                          <Text style={styles.resultCategory} numberOfLines={1}>
                            {formatCategory(item)}
                          </Text>
                        </View>
                        <View style={styles.infoGridItem}>
                          <Text style={styles.infoLabel}>가격</Text>
                          <Text style={styles.resultPrice}>{formatPrice(item.price)}</Text>
                        </View>
                      </View>
                      {similarityText && (
                        <>
                          <View style={styles.infoDivider} />
                          <View style={styles.resultFooterRow}>
                            <Text style={styles.resultSimilarity}>{similarityText}</Text>
                            <TouchableOpacity
                              style={styles.linkButton}
                              onPress={() => openShopLink(item.shop_link)}>
                              <Text style={styles.linkButtonText}>상품 보러가기</Text>
                            </TouchableOpacity>
                          </View>
                        </>
                      )}
                      {!similarityText && (
                        <>
                          <View style={styles.infoDivider} />
                          <TouchableOpacity
                            style={styles.linkButton}
                            onPress={() => openShopLink(item.shop_link)}>
                            <Text style={styles.linkButtonText}>상품 보러가기</Text>
                          </TouchableOpacity>
                        </>
                      )}
                    </View>
                  </View>
                );
              })}

              <TouchableOpacity style={styles.bottomReturnButton} onPress={resetResults}>
                <Text style={styles.primaryButtonText}>검색 화면으로 돌아가기</Text>
              </TouchableOpacity>
            </>
          )}
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.entryContent} keyboardShouldPersistTaps="handled">
        <View style={styles.headerBlock}>
          <Text style={styles.screenTitle} numberOfLines={1} adjustsFontSizeToFit minimumFontScale={0.82}>
            어떤 옷을 찾고 있나요?
          </Text>
          <Text style={styles.subtitle}>사진을 고르고 원하는 조건을 짧게 적어보세요.</Text>
        </View>

        <TouchableOpacity style={styles.imageContainer} onPress={pickImage} activeOpacity={0.85}>
          {imageUri ? (
            <>
              <Image source={{ uri: imageUri }} style={styles.imagePreview} />
              <View style={styles.changeImageBadge}>
                <Text style={styles.changeImageText}>사진 변경</Text>
              </View>
            </>
          ) : (
            <View style={styles.imagePlaceholder}>
              <Ionicons name="image-outline" size={42} color="#5C5E62" />
              <Text style={styles.imagePlaceholderTitle}>옷 사진 선택</Text>
              <Text style={styles.imagePlaceholderBody}>비슷한 상품을 찾을 기준 이미지를 올려주세요.</Text>
            </View>
          )}
        </TouchableOpacity>

        <TextInput
          style={styles.textInput}
          placeholder="예: 회색 와이드 데님 팬츠"
          placeholderTextColor="#8E8E8E"
          value={searchText}
          onChangeText={setSearchText}
          returnKeyType="search"
        />

        {errorMessage ? (
          <Text style={styles.errorText}>{errorMessage}</Text>
        ) : (
          <Text style={styles.helperText}>검색어는 선택 사항입니다. 사진만으로도 검색할 수 있어요.</Text>
        )}

        <View style={styles.knobPanel}>
          <Text style={styles.knobPanelTitle}>🧪 v3 테스트 노브 (비우면 서버 기본값)</Text>
          <View style={styles.knobRow}>
            <View style={styles.knobField}>
              <Text style={styles.knobLabel}>image_weight (w2)</Text>
              <TextInput
                style={styles.knobInput}
                placeholder="0.5"
                placeholderTextColor="#8E8E8E"
                value={imageWeightInput}
                onChangeText={setImageWeightInput}
                keyboardType="decimal-pad"
              />
            </View>
            <View style={styles.knobField}>
              <Text style={styles.knobLabel}>score_threshold</Text>
              <TextInput
                style={styles.knobInput}
                placeholder="0.15"
                placeholderTextColor="#8E8E8E"
                value={scoreThresholdInput}
                onChangeText={setScoreThresholdInput}
                keyboardType="decimal-pad"
              />
            </View>
          </View>
          <View style={styles.knobRow}>
            <View style={styles.knobField}>
              <Text style={styles.knobLabel}>color_penalty_scale</Text>
              <TextInput
                style={styles.knobInput}
                placeholder="2.0"
                placeholderTextColor="#8E8E8E"
                value={colorPenaltyInput}
                onChangeText={setColorPenaltyInput}
                keyboardType="decimal-pad"
              />
            </View>
            <TouchableOpacity
              style={styles.gateToggle}
              onPress={() => setChangeGate(prev => !prev)}
              activeOpacity={0.7}>
              <View style={[styles.checkbox, changeGate && styles.checkboxChecked]}>
                {changeGate && <Ionicons name="checkmark" size={15} color="#FFFFFF" />}
              </View>
              <Text style={styles.gateToggleLabel}>change-gate{'\n'}(색 교체 시 w2=0)</Text>
            </TouchableOpacity>
          </View>
        </View>

        <TouchableOpacity
          style={[styles.primaryButton, isLoading && styles.disabledButton]}
          onPress={() => searchClothes('v1')}
          disabled={isLoading}>
          {isLoading && loadingVersion === 'v1' ? (
            <View style={styles.loadingRow}>
              <ActivityIndicator color="#fff" />
              <Text style={styles.primaryButtonText}>찾는 중...</Text>
            </View>
          ) : (
            <Text style={styles.primaryButtonText}>비슷한 옷 찾기 (v1)</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.v3Button, isLoading && styles.disabledButton]}
          onPress={() => searchClothes('v3')}
          disabled={isLoading}>
          {isLoading && loadingVersion === 'v3' ? (
            <View style={styles.loadingRow}>
              <ActivityIndicator color="#fff" />
              <Text style={styles.primaryButtonText}>찾는 중...</Text>
            </View>
          ) : (
            <Text style={styles.primaryButtonText}>🧪 v3로 검색 (하이브리드)</Text>
          )}
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 28,
    backgroundColor: '#FFFFFF',
  },
  entryContent: {
    flexGrow: 1,
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 32,
    paddingBottom: 32,
  },
  resultContent: {
    paddingHorizontal: 20,
    paddingTop: 28,
    paddingBottom: 32,
  },
  headerBlock: {
    width: '100%',
    maxWidth: 430,
    alignItems: 'center',
    marginBottom: 24,
  },
  brandTitle: {
    fontSize: 32,
    fontWeight: '600',
    color: '#171A20',
    marginBottom: 10,
  },
  screenTitle: {
    fontSize: 26,
    fontWeight: '600',
    color: '#171A20',
    textAlign: 'center',
  },
  subtitle: {
    marginTop: 8,
    fontSize: 16,
    lineHeight: 23,
    color: '#393C41',
    textAlign: 'center',
    width: '100%',
  },
  helperText: {
    marginTop: 12,
    width: '100%',
    maxWidth: 360,
    fontSize: 13,
    lineHeight: 19,
    color: '#5C5E62',
    textAlign: 'center',
  },
  errorText: {
    marginTop: 12,
    fontSize: 13,
    lineHeight: 19,
    color: '#B42318',
  },
  imageContainer: {
    width: '100%',
    maxWidth: 430,
    height: 270,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#F4F4F4',
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  imagePlaceholder: {
    alignItems: 'center',
    paddingHorizontal: 28,
  },
  imagePlaceholderTitle: {
    marginTop: 12,
    fontSize: 17,
    fontWeight: '600',
    color: '#171A20',
  },
  imagePlaceholderBody: {
    marginTop: 6,
    fontSize: 14,
    lineHeight: 20,
    color: '#5C5E62',
    textAlign: 'center',
  },
  imagePreview: {
    width: '100%',
    height: '100%',
  },
  changeImageBadge: {
    position: 'absolute',
    right: 12,
    bottom: 12,
    borderRadius: 4,
    backgroundColor: 'rgba(255,255,255,0.92)',
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  changeImageText: {
    color: '#171A20',
    fontSize: 13,
    fontWeight: '600',
  },
  textInput: {
    width: '100%',
    maxWidth: 430,
    height: 56,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#FFFFFF',
    color: '#171A20',
    paddingHorizontal: 14,
    paddingVertical: 14,
    fontSize: 15,
    lineHeight: 21,
    textAlignVertical: 'center',
  },
  primaryButton: {
    width: '100%',
    maxWidth: 430,
    minHeight: 52,
    borderRadius: 6,
    backgroundColor: '#3E6AE1',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
    marginTop: 20,
  },
  disabledButton: {
    opacity: 0.72,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  v3Button: {
    width: '100%',
    maxWidth: 430,
    minHeight: 52,
    borderRadius: 6,
    backgroundColor: '#16A34A',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
    marginTop: 10,
  },
  knobPanel: {
    width: '100%',
    maxWidth: 430,
    marginTop: 14,
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    backgroundColor: '#F8FAFB',
  },
  knobPanelTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#393C41',
    marginBottom: 10,
  },
  knobRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 10,
  },
  knobField: {
    flex: 1,
  },
  knobLabel: {
    fontSize: 11,
    color: '#5C5E62',
    marginBottom: 4,
  },
  knobInput: {
    height: 42,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E4E7EC',
    backgroundColor: '#FFFFFF',
    color: '#171A20',
    paddingHorizontal: 12,
    fontSize: 14,
  },
  gateToggle: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  gateToggleLabel: {
    flex: 1,
    fontSize: 11,
    color: '#393C41',
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 4,
    borderWidth: 1.5,
    borderColor: '#16A34A',
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxChecked: {
    backgroundColor: '#16A34A',
  },
  resultHeader: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  resultHeaderRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginTop: 6,
  },
  versionBadge: {
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 12,
  },
  versionBadgeV1: {
    backgroundColor: '#EBEEF5',
  },
  versionBadgeV3: {
    backgroundColor: '#D7F2E5',
  },
  versionBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#393C41',
  },
  resultCount: {
    color: '#5C5E62',
    fontSize: 14,
    textAlign: 'center',
  },
  emptyState: {
    minHeight: 420,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
    backgroundColor: '#F4F4F4',
    paddingHorizontal: 24,
  },
  emptyTitle: {
    marginTop: 16,
    color: '#171A20',
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
  },
  emptyBody: {
    marginTop: 8,
    color: '#5C5E62',
    fontSize: 14,
    lineHeight: 20,
    textAlign: 'center',
  },
  resultCard: {
    flexDirection: 'row',
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#EEEEEE',
    backgroundColor: '#FFFFFF',
    padding: 12,
    marginBottom: 14,
  },
  resultImage: {
    width: 116,
    height: 148,
    borderRadius: 6,
    marginRight: 14,
    backgroundColor: '#F4F4F4',
  },
  resultInfo: {
    flex: 1,
    minHeight: 148,
  },
  infoSection: {
    paddingBottom: 7,
  },
  infoDivider: {
    height: 1,
    backgroundColor: '#EEEEEE',
    marginBottom: 7,
  },
  infoLabel: {
    color: '#8E8E8E',
    fontSize: 11,
    lineHeight: 15,
    marginBottom: 2,
  },
  infoGrid: {
    flexDirection: 'row',
    gap: 10,
    paddingBottom: 7,
  },
  infoGridItem: {
    flex: 1,
  },
  resultBrand: {
    color: '#393C41',
    fontSize: 14,
    fontWeight: '600',
  },
  resultName: {
    color: '#171A20',
    fontSize: 15,
    fontWeight: '600',
    lineHeight: 20,
  },
  resultCategory: {
    color: '#5C5E62',
    fontSize: 12,
    lineHeight: 16,
  },
  resultPrice: {
    color: '#171A20',
    fontSize: 15,
    fontWeight: '600',
  },
  resultSimilarity: {
    color: '#5C5E62',
    fontSize: 12,
    lineHeight: 18,
  },
  resultFooterRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  linkButton: {
    alignSelf: 'flex-start',
    justifyContent: 'center',
  },
  linkButtonText: {
    color: '#3E6AE1',
    fontSize: 14,
    fontWeight: '600',
  },
  bottomReturnButton: {
    width: '100%',
    maxWidth: 430,
    minHeight: 52,
    borderRadius: 6,
    backgroundColor: '#3E6AE1',
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
    marginTop: 10,
    marginBottom: 20,
  },
});
