import { Alert, AlertButton, Platform } from 'react-native';

// Alert.alert는 웹 브라우저에서 아무 창도 띄우지 않는다.
// 웹에서는 브라우저 알림창(window.alert)을 쓰고, 앱(iOS/Android)에서는 기존 Alert.alert와 똑같이 동작한다.
// 웹에서는 버튼을 고를 수 없어서, 버튼이 하나일 때 그 버튼의 onPress를 창을 닫은 뒤 실행한다.
export function notify(title: string, message?: string, buttons?: AlertButton[]) {
  if (Platform.OS === 'web') {
    window.alert(message ? `${title}\n\n${message}` : title);
    if (buttons && buttons.length === 1) {
      buttons[0].onPress?.();
    }
    return;
  }
  Alert.alert(title, message, buttons);
}
