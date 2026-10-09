import { FontAwesome, Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { notify } from '../../lib/notify';

export default function LoginScreen() {
  const [session, setSession] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showEmailForm, setShowEmailForm] = useState(false);
  const [nickname, setNickname] = useState('');

  // 앱 시작 시 저장된 토큰이 있는지 확인
  useEffect(() => {
    checkToken();
  }, []);

  const checkToken = async () => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      const savedEmail = await AsyncStorage.getItem('userEmail');
      const savedNickname = await AsyncStorage.getItem('userNickname');
      
      if (token) {
        if (savedEmail) setEmail(savedEmail);
        if (savedNickname) setNickname(savedNickname);
        setSession(true);
      }
    } catch (e) {
      console.log('토큰 확인 실패', e);
    }
  };

  const signInWithEmail = async () => {
    if (!email.trim() || !password) {
      notify('입력 필요', '이메일과 비밀번호를 입력해주세요.');
      return;
    }

    setLoading(true);
    try {
      const API_URL = process.env.EXPO_PUBLIC_FASHION_API_URL?.replace(/\/$/, '') || 'http://localhost:8080/api';
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: email.trim(),
          password: password,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // 스프링 부트에서 넘겨준 토큰 저장
        await AsyncStorage.setItem('userToken', data.accessToken);
        await AsyncStorage.setItem('userEmail', data.email || email);
        if (data.nickname) await AsyncStorage.setItem('userNickname', data.nickname);
        
        setNickname(data.nickname || '회원');
        setSession(true);
      } else {
        const err = await response.json();
        notify('로그인 실패', err.error || '이메일이나 비밀번호가 틀렸습니다.');
      }
    } catch (error) {
      notify('네트워크 에러', '서버에 연결할 수 없습니다.');
    }
    setLoading(false);
  };

  const signOut = async () => {
    await AsyncStorage.removeItem('userToken');
    await AsyncStorage.removeItem('userEmail');
    await AsyncStorage.removeItem('userNickname');
    setSession(false);
    setPassword('');
    setShowEmailForm(false);
  };

  const showDeleteAccountNotice = () => {
    notify('회원탈퇴', '회원탈퇴 기능은 현재 구현 중입니다.');
  };

  if (!session) {
    if (showEmailForm) {
      return (
        <SafeAreaView style={styles.safeArea}>
          <KeyboardAvoidingView style={styles.keyboardContainer} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
            <ScrollView contentContainerStyle={styles.emailScreenContent} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
              <TouchableOpacity style={styles.backIconButton} onPress={() => setShowEmailForm(false)} activeOpacity={0.72}>
                <Ionicons name="chevron-back" size={28} color="#171A20" />
              </TouchableOpacity>
              <View style={styles.emailScreenHeader}>
                <Text style={styles.emailScreenTitle}>이메일로 로그인</Text>
                <Text style={styles.emailScreenSubtitle}>가입한 이메일과 비밀번호를 입력해주세요.</Text>
              </View>
              <View style={styles.emailForm}>
                <TextInput style={styles.input} placeholder="이메일 주소" placeholderTextColor="#A0A0A0" value={email} onChangeText={setEmail} autoCapitalize="none" keyboardType="email-address" editable={!loading} />
                <TextInput style={styles.input} placeholder="비밀번호" placeholderTextColor="#A0A0A0" value={password} onChangeText={setPassword} secureTextEntry autoCapitalize="none" editable={!loading} />
                <TouchableOpacity style={[styles.primaryButton, loading && styles.disabledButton]} onPress={signInWithEmail} disabled={loading} activeOpacity={0.8}>
                  {loading ? <ActivityIndicator color="#FFFFFF" /> : <Text style={styles.primaryButtonText}>로그인</Text>}
                </TouchableOpacity>
              </View>
            </ScrollView>
          </KeyboardAvoidingView>
        </SafeAreaView>
      );
    }
    return (
      <SafeAreaView style={styles.safeArea}>
        <ScrollView contentContainerStyle={styles.authContent} bounces={false}>
          <View style={styles.guestHero}>
            <Text style={styles.guestTitle}>나만의 패션을{'\n'}완성해보세요.</Text>
            <TouchableOpacity style={styles.guestLoginButton} onPress={() => setShowEmailForm(true)} activeOpacity={0.8}>
              <Text style={styles.guestLoginButtonText}>이메일로 로그인</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.actionBlock}>
            <TouchableOpacity style={styles.signUpInlineButton} onPress={() => router.push('/signup')} activeOpacity={0.72}>
              <Text style={styles.signUpInlineText}>아직 계정이 없으신가요? 회원가입</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.guestMenu}>
            <Text style={styles.kicker}>고객 지원</Text>
            <TouchableOpacity style={styles.guestMenuRow} activeOpacity={0.72}>
              <Text style={styles.guestMenuText}>공지사항</Text>
              <Ionicons name="chevron-forward" size={20} color="#D0D1D2" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.guestMenuRow} activeOpacity={0.72}>
              <Text style={styles.guestMenuText}>이용 가이드</Text>
              <Ionicons name="chevron-forward" size={20} color="#D0D1D2" />
            </TouchableOpacity>
          </View>
        </ScrollView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.profileContent}>
        <View style={styles.memberHero}>
          <Text style={styles.memberName}>{nickname}님, 반가워요!</Text>
          <View style={styles.memberSummaryCard}>
            <View style={styles.memberAvatarBox}>
              <Ionicons name="person-circle-outline" size={52} color="#C0C0C0" />
              <Text style={styles.memberAvatarLabel}>내 프로필</Text>
            </View>
            <View style={styles.memberInfoBox}>
              <View style={styles.memberInfoRow}>
                <Text style={styles.memberInfoLabel}>이메일</Text>
                <Text style={styles.memberInfoValue}>{email}</Text>
              </View>
              <View style={styles.memberDivider} />
              <View style={styles.memberInfoRow}>
                <Text style={styles.memberInfoLabel}>가입 방식</Text>
                <Text style={styles.memberInfoValue}>이메일 (Spring Boot)</Text>
              </View>
            </View>
          </View>
        </View>
        <View style={styles.memberMenu}>
          <TouchableOpacity style={styles.memberMenuRow} onPress={signOut} activeOpacity={0.72}>
            <Text style={styles.memberMenuText}>로그아웃</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.memberMenuRow} onPress={showDeleteAccountNotice} activeOpacity={0.72}>
            <Text style={styles.deleteMenuText}>회원탈퇴</Text>
            <Ionicons name="chevron-forward" size={18} color="#D0D1D2" />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#FFFFFF' },
  keyboardContainer: { flex: 1 },
  authContent: { flexGrow: 1, paddingTop: 82, paddingBottom: 36 },
  guestHero: { paddingHorizontal: 28, paddingBottom: 38, borderBottomWidth: 8, borderBottomColor: '#F4F4F4' },
  guestTitle: { color: '#171A20', fontSize: 26, lineHeight: 36, fontWeight: '600', marginBottom: 28 },
  guestLoginButton: { minHeight: 62, borderRadius: 8, backgroundColor: '#3E6AE1', alignItems: 'center', justifyContent: 'center' },
  guestLoginButtonText: { color: '#FFFFFF', fontSize: 17, lineHeight: 23, fontWeight: '600' },
  actionBlock: { width: '100%', maxWidth: 430, alignSelf: 'center', gap: 12, paddingHorizontal: 28, paddingTop: 22 },
  signUpInlineButton: { minHeight: 44, alignItems: 'center', justifyContent: 'center', borderRadius: 6, backgroundColor: '#F4F4F4' },
  signUpInlineText: { color: '#393C41', fontSize: 14, lineHeight: 19, fontWeight: '600' },
  emailScreenContent: { flexGrow: 1, paddingHorizontal: 28, paddingTop: 18, paddingBottom: 36 },
  backIconButton: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center', marginLeft: -12 },
  emailScreenHeader: { marginTop: 52, marginBottom: 28 },
  emailScreenTitle: { color: '#171A20', fontSize: 30, lineHeight: 38, fontWeight: '600' },
  emailScreenSubtitle: { marginTop: 10, color: '#5C5E62', fontSize: 14, lineHeight: 20 },
  emailForm: { gap: 10, marginTop: 2 },
  input: { minHeight: 52, backgroundColor: '#FFFFFF', paddingHorizontal: 14, borderRadius: 6, fontSize: 15, borderWidth: 1, borderColor: '#EEEEEE', color: '#171A20' },
  primaryButton: { minHeight: 52, borderRadius: 6, backgroundColor: '#3E6AE1', alignItems: 'center', justifyContent: 'center', marginTop: 2 },
  primaryButtonText: { fontSize: 16, fontWeight: '600', color: '#FFFFFF' },
  disabledButton: { opacity: 0.68 },
  guestMenu: { paddingHorizontal: 28, paddingTop: 28 },
  guestMenuRow: { minHeight: 64, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  guestMenuText: { color: '#171A20', fontSize: 19, lineHeight: 25, fontWeight: '600' },
  kicker: { color: '#3E6AE1', fontSize: 22, lineHeight: 28, fontWeight: '600', marginBottom: 18 },
  profileContent: { flexGrow: 1, backgroundColor: '#FFFFFF', paddingBottom: 42 },
  memberHero: { paddingHorizontal: 24, paddingTop: 86, paddingBottom: 24, backgroundColor: '#F4F4F4' },
  memberName: { color: '#171A20', fontSize: 31, lineHeight: 39, fontWeight: '600', marginBottom: 20 },
  memberSummaryCard: { flexDirection: 'row', gap: 12 },
  memberAvatarBox: { width: 122, minHeight: 128, borderRadius: 8, backgroundColor: '#FFFFFF', alignItems: 'center', justifyContent: 'center', paddingHorizontal: 12 },
  memberAvatarLabel: { color: '#8E8E8E', fontSize: 13, lineHeight: 19, fontWeight: '600', marginTop: 10 },
  memberInfoBox: { flex: 1, minHeight: 128, borderRadius: 8, backgroundColor: '#FFFFFF', paddingHorizontal: 16, justifyContent: 'center' },
  memberInfoRow: { minHeight: 48, justifyContent: 'center' },
  memberInfoLabel: { color: '#8E8E8E', fontSize: 10, lineHeight: 15, marginBottom: 3 },
  memberInfoValue: { color: '#171A20', fontSize: 13, lineHeight: 19, fontWeight: '600' },
  memberDivider: { height: 1, backgroundColor: '#EEEEEE' },
  memberMenu: { paddingHorizontal: 24, paddingVertical: 18, borderBottomWidth: 8, borderBottomColor: '#F4F4F4' },
  memberMenuRow: { minHeight: 58, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  memberMenuText: { color: '#171A20', fontSize: 18, lineHeight: 25, fontWeight: '600' },
  deleteMenuText: { color: '#B42318', fontSize: 18, lineHeight: 25, fontWeight: '600' },
});
