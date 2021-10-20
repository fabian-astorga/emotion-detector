import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import * as ImagePicker from 'expo-image-picker';
import { api } from './config';
import { 
  ImageBackground, 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity,
  Alert, 
  BackHandler,
  Image 
} from 'react-native';

const sendImage = async (base64, { navigation }) => {

  fetch(api.URL + 'process-image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      "image": base64
    }),
  }).then((response) => response.json())
      .then((responseJson) => {
        navigation.navigate('Resultado', 
        {
          finalImage: responseJson.response, 
          tags: responseJson.tags
        });
      })
      .catch((error) => {
        console.log('ERROR: ' + error);
        Alert.alert('Algo salió mal. Intentalo de nuevo.');
      });
};

const openGallery = async ({ navigation }) => {
  let result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.All,
    allowsEditing: true,
    aspect: [4, 3],
    quality: 1,
    base64: true
  });

  sendImage(result['base64'], { navigation });
};

const openCamera = async ({ navigation }) => {
  let result = await ImagePicker.launchCameraAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.All,
    allowsEditing: true,
    aspect: [4, 3],
    quality: 1,
    base64: true
  });

  sendImage(result['base64'], { navigation });
};

function HomeScreen({ navigation }) { 

  return (
    <View style={styles.container}>
      <ImageBackground source={require('./images/background.jpg')} resizeMode="cover" style={styles.image}>
        <Text style={styles.text}>RECONOCEDOR DE EMOCIONES</Text>
        <TouchableOpacity onPress={() => openGallery({ navigation })} style={styles.button}>
          <Text style={styles.buttonText}>CARGAR IMAGEN</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => openCamera({ navigation })} style={styles.button}>
          <Text style={styles.buttonText}>ABRIR CÁMARA</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => BackHandler.exitApp()} style={styles.button}>
          <Text style={styles.buttonText}>SALIR</Text>
        </TouchableOpacity>
      </ImageBackground>
    </View>    
  );
}

function ResultScreen({ route }) {

  const { finalImage, tags } = route.params;
  let result = finalImage.split("'");
  //console.log('FINAL IMAGE: ' + result[1]);

  return (
    <View style={styles.container}>
      <ImageBackground source={require('./images/background.jpg')} resizeMode="cover" style={styles.image}>
        <Image style={{width: 400, height: 400, resizeMode: 'cover', borderWidth: 5, borderColor: 'black'}} 
              source={{uri: 'data:image/jpg;base64,'+result[1]}} 
        />
        <Text style={styles.text}>{ tags }</Text>
      </ImageBackground>
    </View>
  )
}

const Stack = createNativeStackNavigator();

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Resultado" component={ResultScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  image: {
    flex: 1,
    justifyContent: "center"
  },
  text: {
    color: "white",
    fontSize: 20,
    lineHeight: 90,
    fontWeight: "bold",
    textAlign: "center",
    backgroundColor: "#000000c0"
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    lineHeight: 40,
    fontWeight: "bold",
    textAlign: "center"
  },
  button: {
    width: 250,
    marginTop: 10,
    borderColor: "black",
    borderWidth: 5,
    backgroundColor: "#ff2e00",
    padding: 10,
    borderRadius: 50,
    alignSelf: "center"
  }
});

export default App;