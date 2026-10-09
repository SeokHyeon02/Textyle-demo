import { router } from 'expo-router';
import React, { useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { notify } from '../lib/notify';

export default function SignUpScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [loading, setLoading] = useState(false);
  const [nickname, setNickname] = useState('');

  const handleSignUp = async () => {
    console.log('🔥 handleSignUp 실행됨');

    setLoading(true);
    if (!email || !password || !nickname) {
      notify('알림', '모든 정보를 입력해주세요.');
      return;
    }
    if (password !== passwordConfirm) {
      notify('알림', '비밀번호가 서로 일치하지 않습니다.');
      return;
    }

    setLoading(true);
    
    try {
      const API_URL = process.env.EXPO_PUBLIC_FASHION_API_URL?.replace(/\/$/, '') || 'http://localhost:8080';

      console.log('회원가입 URL:', `${API_URL}/api/auth/join`);
      const response = await fetch(`${API_URL}/api/auth/join`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email.trim(),
          password: password,
          nickname: nickname.trim(),
        }),
      });

      console.log('회원가입 status:', response.status);

    const responseText = await response.text();
    console.log('회원가입 response:', responseText);

      if (response.ok) {
        notify('가입 성공!', '환영합니다. 이제 로그인해주세요.', [
          { text: '확인', onPress: () => router.replace('/login') },
        ]);
      } else {
        const errorData = await response.json();
        notify('회원가입 실패', errorData.error || '가입 중 오류가 발생했습니다.');
      }
    } catch (e) {
      notify('네트워크 오류', '서버와 통신할 수 없습니다.');
    }
    
    setLoading(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>새 계정 만들기</Text>
      <Text style={styles.subtitle}>TexTyle에 오신 것을 환영합니다</Text>

      <TextInput
        style={styles.input}
        placeholder="이메일"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        keyboardType="email-address"
      />
      <TextInput
        style={styles.input}
        placeholder="닉네임"
        value={nickname}
        onChangeText={setNickname}
      />
      <TextInput
        style={styles.input}
        placeholder="비밀번호 (6자리 이상)"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        autoCapitalize="none"
      />
      <TextInput
        style={styles.input}
        placeholder="비밀번호 확인"
        value={passwordConfirm}
        onChangeText={setPasswordConfirm}
        secureTextEntry
        autoCapitalize="none"
      />

      <TouchableOpacity style={styles.button} onPress={handleSignUp} disabled={loading}>
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>가입하기</Text>
        )}
      </TouchableOpacity>

      <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
        <Text style={styles.backButtonText}>이미 계정이 있으신가요? 로그인</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: '#fff', justifyContent: 'center' },
  title: { fontSize: 28, fontWeight: '600', marginBottom: 10, color: '#171A20' },
  subtitle: { fontSize: 16, color: '#393C41', marginBottom: 30 },
  input: { backgroundColor: '#FFFFFF', padding: 15, borderRadius: 8, marginBottom: 15, fontSize: 16, borderWidth: 1, borderColor: '#EEEEEE', color: '#171A20' },
  button: { backgroundColor: '#3E6AE1', paddingVertical: 15, borderRadius: 6, alignItems: 'center', marginTop: 10 },
  buttonText: { fontSize: 16, fontWeight: '600', color: '#fff' },
  backButton: { marginTop: 20, alignItems: 'center' },
  backButtonText: { color: '#3E6AE1', fontSize: 15, fontWeight: '500' }
});
