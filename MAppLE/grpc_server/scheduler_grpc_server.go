package main

import (
	"fmt"
	"log"
	"net"

	"google.golang.org/grpc"
	"github.com/BeIanChang/SAC/quic"
)

func main() {
	// Start gRPC server
	listener, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	quic.RegisterMPQUICServiceServer(grpcServer, &quic.MPQUICServer{})

	fmt.Println("Server is running on port 50051...")
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
