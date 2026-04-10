import { FontAwesome } from '@expo/vector-icons';
import * as Google from 'expo-auth-session/providers/google';
import { router } from 'expo-router';
import * as WebBrowser from 'expo-web-browser';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { supabase } from '../../supabase';

WebBrowser.maybeCompleteAuthSession();

export default function LoginScreen() {
  const [session, setSession] = useState<any>(null);
  
  // 👈 이메일, 비밀번호 입력을 위한 상태 추가
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  // 구글 로그인 설정 (이전에 발급받은 클라이언트 ID를 넣어주세요)
  const [request, response, promptAsync] = Google.useAuthRequest({
    webClientId: '509294193303-6fc0fgvftk04hb7l0frqta6lmmejdoop.apps.googleusercontent.com',
    iosClientId: '509294193303-km6ho5gcvu02cfhiqlurc2dbppindte3.apps.googleusercontent.com', 
  });

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    supabase.auth.onAuthStateChange((_event, session) => setSession(session));
  }, []);

  // 구글 로그인 처리
  useEffect(() => {
    if (response?.type === 'success') {
      const { id_token } = response.params;
      supabase.auth.signInWithIdToken({ provider: 'google', token: id_token })
        .then(({ error }) => {
          if (error) Alert.alert('오류', error.message);
        });
    }
  }, [response]);

  // ✉️ 일반 이메일 로그인 함수
  const signInWithEmail = async () => {
    setLoading(true);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) Alert.alert('로그인 실패', error.message);
    setLoading(false);
  };


  const signOut = async () => {
    await supabase.auth.signOut();
  };

  // ------------------------------------------------------------------
  // 로그인 안 된 상태 화면 (이메일 폼 + 구글 로그인)
  // ------------------------------------------------------------------
  if (!session) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>TexTyle AI</Text>
        <Text style={styles.subtitle}>이메일 또는 구글로 시작하세요</Text>

        {/* 이메일/비밀번호 입력창 */}
        <TextInput
          style={styles.input}
          placeholder="이메일 주소"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />
        <TextInput
          style={styles.input}
          placeholder="비밀번호 (6자리 이상)"
          value={password}
          onChangeText={setPassword}
          secureTextEntry // 비밀번호 가리기
          autoCapitalize="none"
        />

        {/* 이메일 로그인 & 가입 버튼 */}
        <View style={styles.buttonRow}>
          <TouchableOpacity style={[styles.button, styles.signInBtn]} onPress={signInWithEmail} disabled={loading}>
            <Text style={styles.buttonText}>로그인</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.button, styles.signUpBtn]} onPress={() => router.push('/signup')} disabled={loading}>
            <Text style={[styles.buttonText, styles.signUpText]}>회원가입</Text>
          </TouchableOpacity>
        </View>

        {loading && <ActivityIndicator size="large" color="#8A2BE2" style={{ marginVertical: 20 }} />}

        <View style={styles.divider}>
          <View style={styles.line} />
          <Text style={styles.orText}>또는</Text>
          <View style={styles.line} />
        </View>
        
        {/* 구글 로그인 버튼 */}
        <TouchableOpacity style={styles.googleButton} onPress={() => promptAsync()} disabled={!request}>
          <FontAwesome name="google" size={20} color="#fff" style={styles.icon} />
          <Text style={styles.googleButtonText}>Google로 시작하기</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ------------------------------------------------------------------
  // 로그인 된 상태 (마이페이지)
  // ------------------------------------------------------------------
  return (
    <View style={styles.container}>
      <Text style={styles.title}>마이페이지</Text>
      <View style={styles.profileCard}>
        <FontAwesome name="user-circle" size={80} color="#ddd" />
        <Text style={styles.email}>{session.user.email}</Text>
      </View>
      <TouchableOpacity style={styles.logoutButton} onPress={signOut}>
        <Text style={styles.logoutButtonText}>로그아웃</Text>
      </TouchableOpacity>
    </View>
  );
}

// 스타일
const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: '#fff', justifyContent: 'center' },
  title: { fontSize: 28, fontWeight: 'bold', marginBottom: 5, textAlign: 'center', color: '#333' },
  subtitle: { fontSize: 16, color: '#666', marginBottom: 30, textAlign: 'center' },
  
  input: { backgroundColor: '#f5f5f5', padding: 15, borderRadius: 10, marginBottom: 15, fontSize: 16, borderWidth: 1, borderColor: '#eee' },
  buttonRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20 },
  button: { paddingVertical: 15, borderRadius: 10, width: '48%', alignItems: 'center' },
  signInBtn: { backgroundColor: '#8A2BE2' },
  signUpBtn: { backgroundColor: '#fff', borderWidth: 1, borderColor: '#8A2BE2' },
  signUpText: { color: '#8A2BE2' },
  buttonText: { fontSize: 16, fontWeight: 'bold', color: '#fff' },
  
  divider: { flexDirection: 'row', alignItems: 'center', marginVertical: 20 },
  line: { flex: 1, height: 1, backgroundColor: '#eee' },
  orText: { marginHorizontal: 10, color: '#888' },

  googleButton: { flexDirection: 'row', backgroundColor: '#4285F4', padding: 15, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  icon: { marginRight: 15 },
  googleButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },

  profileCard: { backgroundColor: '#f9f9f9', padding: 30, borderRadius: 15, alignItems: 'center', marginBottom: 40, borderWidth: 1, borderColor: '#eee' },
  email: { fontSize: 18, color: '#555', marginTop: 15 },
  logoutButton: { backgroundColor: '#eee', paddingVertical: 15, borderRadius: 10, alignItems: 'center' },
  logoutButtonText: { color: '#333', fontSize: 16, fontWeight: 'bold' }
});