pipeline {
  agent any
  stages {
    stage('Build and Test') {
      steps {
        sh '''export PYTHONPREFIX=$WORKSPACE
export PYTHONPATH=$PYTHONPREFIX/lib/python3.6/site-packages

mkdir -p $WORKSPACE/lib/python3.6/site-packages

make test'''
      }
    }
  }
}