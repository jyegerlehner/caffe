#include <fstream>
#include <vector>
#include <list>
#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <time.h>

int main(int argc, char** argv)
{
  srand(time(NULL));
  std::cout << "RAND_MAX==" << RAND_MAX << std::endl;
  std::ifstream fs;
  fs.open( "fall11_urls.txt" );
  if ( !fs.is_open())
  {
    throw std::runtime_error( "failed to open urls file." );
  }
  std::string line;
  std::vector<std::string> urls;
  int line_ctr = 0;
  while ( getline( fs, line ) )
  {
    std::vector<std::string> tokens;
    boost::split(tokens, line, boost::is_any_of(" \t") );
    if ( tokens.size() == 2 )
    {
      std::string url = tokens[tokens.size()-1];
      urls.push_back(url);
      line_ctr++;
      if ( ( line_ctr % 1000 ) == 0 )
      {
        std::cout << ".";
      }
    }
  }

  fs.close();
  std::cout << std::endl;

  int url_count = urls.size();
  std::cout << "Number of urls:" << url_count << std::endl;

  int ctr = 0;
  while( ctr < 1000 )
  {
    // randomly select an image from the list to retrieve.
    int index = rand() % url_count;
    std::string url = urls[index];
    urls[index] = std::string();

    if ( !url.empty() )
    {
      std::string command = "wget -t 1 -T 30 -nc ";
      command += url;
      if ( 0 == system( command.c_str() ) )
      {
        std::cout << ctr << ", retrieved " << url << std::endl;
        ctr++;
      }
      else
      {
        std::cout << ctr << ", failed to retrieve " << url << std::endl;
      }
    }
  }
  return 0;
}
